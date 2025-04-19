"""
Implementación del Optimizador Automático para NEBULA.

Esta clase proporciona mecanismos para la automejora del sistema NEBULA,
permitiendo optimizar su propio código y configuración.
"""

import logging
import time
import os
import re
import ast
import importlib
import sys
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from pathlib import Path
import json
import tempfile

from utils.config import PARAMETERS
from utils.helpers import safe_loop

logger = logging.getLogger("NEBULA.SelfOptimizer")

class SelfOptimizer:
    """
    Optimizador automático que permite a NEBULA mejorar su propio código
    y configuración de forma autónoma.
    
    Características:
    - Optimización de parámetros de configuración
    - Refactorización automática de código
    - Implementación de mejoras sugeridas
    - Pruebas automáticas de cambios
    - Gestión de versiones y rollback
    """
    
    def __init__(self, evolution_engine=None, code_analyzer=None, llm_manager=None):
        """
        Inicializa el optimizador automático.
        
        Args:
            evolution_engine: Motor de evolución para optimización de parámetros.
            code_analyzer: Analizador de código para evaluación y sugerencias.
            llm_manager: Gestor de modelos LLM para generación de código.
        """
        logger.info("Inicializando SelfOptimizer...")
        self.evolution_engine = evolution_engine
        self.code_analyzer = code_analyzer
        self.llm_manager = llm_manager
        
        # Directorio raíz del proyecto
        self.project_root = Path(PARAMETERS["PROJECT_ROOT"])
        
        # Directorio para backups y versiones
        self.versions_dir = PARAMETERS["BACKUP_DIRECTORY"] / "versions"
        os.makedirs(self.versions_dir, exist_ok=True)
        
        # Historial de optimizaciones
        self.optimization_history = []
        
        # Contadores y estadísticas
        self.total_optimizations = 0
        self.successful_optimizations = 0
        self.last_optimization_time = 0
        
        # Configuración de seguridad
        self.safety_checks = {
            "backup_before_changes": True,
            "run_tests_after_changes": True,
            "max_changes_per_run": 5,
            "restricted_modules": ["core", "utils"],  # Módulos con restricciones adicionales
            "allow_structural_changes": False,  # Cambios en estructura de clases/métodos
            "allow_config_changes": True,  # Cambios en parámetros de configuración
        }
        
        logger.info("SelfOptimizer inicializado correctamente.")
    
    def _create_backup(self, target_path: Optional[Path] = None, version_tag: Optional[str] = None) -> Path:
        """
        Crea una copia de seguridad del código actual.
        
        Args:
            target_path: Ruta específica a respaldar (None para todo el proyecto).
            version_tag: Etiqueta de versión (None para timestamp).
            
        Returns:
            Ruta al directorio de backup.
        """
        timestamp = int(time.time())
        version_tag = version_tag or f"backup_{timestamp}"
        
        # Determinar directorio de origen
        source_dir = target_path if target_path else self.project_root
        
        # Crear directorio de backup
        backup_dir = self.versions_dir / version_tag
        os.makedirs(backup_dir, exist_ok=True)
        
        try:
            # Copiar archivos
            if target_path and target_path.is_file():
                # Backup de un solo archivo
                dest_file = backup_dir / target_path.name
                shutil.copy2(target_path, dest_file)
                logger.info(f"Backup creado para archivo {target_path} en {dest_file}")
            else:
                # Backup de directorio
                if target_path:
                    # Copiar solo el directorio específico
                    dest_dir = backup_dir / target_path.name
                    shutil.copytree(target_path, dest_dir)
                    logger.info(f"Backup creado para directorio {target_path} en {dest_dir}")
                else:
                    # Copiar todo el proyecto
                    for item in source_dir.iterdir():
                        if item.is_dir() and not item.name.startswith('.') and item.name != 'versions':
                            dest_dir = backup_dir / item.name
                            shutil.copytree(item, dest_dir)
                        elif item.is_file():
                            dest_file = backup_dir / item.name
                            shutil.copy2(item, dest_file)
                    
                    logger.info(f"Backup completo del proyecto creado en {backup_dir}")
            
            # Crear archivo de metadatos
            metadata = {
                "timestamp": timestamp,
                "version_tag": version_tag,
                "source": str(source_dir),
                "created_by": "SelfOptimizer",
                "description": f"Backup automático antes de optimización"
            }
            
            with open(backup_dir / "backup_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return backup_dir
        
        except Exception as e:
            logger.error(f"Error al crear backup: {e}")
            # Intentar limpiar en caso de error
            if backup_dir.exists():
                try:
                    shutil.rmtree(backup_dir)
                except:
                    pass
            raise
    
    def _restore_from_backup(self, backup_path: Path) -> bool:
        """
        Restaura el código desde una copia de seguridad.
        
        Args:
            backup_path: Ruta al directorio de backup.
            
        Returns:
            True si la restauración fue exitosa, False en caso contrario.
        """
        if not backup_path.exists() or not backup_path.is_dir():
            logger.error(f"Directorio de backup no encontrado: {backup_path}")
            return False
        
        try:
            # Verificar metadata
            metadata_file = backup_path / "backup_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                source = metadata.get("source")
                if source:
                    source_path = Path(source)
                    
                    # Determinar si es backup de archivo, directorio o proyecto completo
                    if source_path.is_file():
                        # Restaurar archivo
                        backup_file = backup_path / source_path.name
                        if backup_file.exists():
                            shutil.copy2(backup_file, source_path)
                            logger.info(f"Archivo {source_path} restaurado desde {backup_file}")
                            return True
                    else:
                        # Restaurar directorio o proyecto
                        for item in backup_path.iterdir():
                            if item.name == "backup_metadata.json":
                                continue
                            
                            if item.is_dir():
                                dest_dir = self.project_root / item.name
                                if dest_dir.exists():
                                    shutil.rmtree(dest_dir)
                                shutil.copytree(item, dest_dir)
                            elif item.is_file():
                                dest_file = self.project_root / item.name
                                shutil.copy2(item, dest_file)
                        
                        logger.info(f"Proyecto restaurado desde backup {backup_path}")
                        return True
            
            # Si no hay metadata o falló la restauración basada en metadata,
            # intentar restauración genérica
            for item in backup_path.iterdir():
                if item.name == "backup_metadata.json":
                    continue
                
                if item.is_dir():
                    dest_dir = self.project_root / item.name
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                    shutil.copytree(item, dest_dir)
                elif item.is_file():
                    dest_file = self.project_root / item.name
                    shutil.copy2(item, dest_file)
            
            logger.info(f"Proyecto restaurado desde backup {backup_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error al restaurar desde backup: {e}")
            return False
    
    def _run_tests(self, target_module: Optional[str] = None) -> Dict[str, Any]:
        """
        Ejecuta pruebas para verificar la integridad del sistema.
        
        Args:
            target_module: Módulo específico a probar (None para todos).
            
        Returns:
            Diccionario con resultados de pruebas.
        """
        start_time = time.time()
        
        # Directorio de pruebas
        tests_dir = self.project_root / "tests"
        
        if not tests_dir.exists():
            logger.warning(f"Directorio de pruebas no encontrado: {tests_dir}")
            return {
                "status": "error",
                "error": "Directorio de pruebas no encontrado",
                "passed": False
            }
        
        try:
            # Determinar pruebas a ejecutar
            if target_module:
                test_pattern = f"test_{target_module}*.py"
                test_files = list(tests_dir.glob(test_pattern))
                
                if not test_files:
                    logger.warning(f"No se encontraron pruebas para el módulo {target_module}")
                    return {
                        "status": "warning",
                        "warning": f"No se encontraron pruebas para el módulo {target_module}",
                        "passed": True  # Asumir éxito si no hay pruebas
                    }
            else:
                test_files = list(tests_dir.glob("test_*.py"))
                
                if not test_files:
                    logger.warning("No se encontraron archivos de prueba")
                    return {
                        "status": "warning",
                        "warning": "No se encontraron archivos de prueba",
                        "passed": True  # Asumir éxito si no hay pruebas
                    }
            
            # Ejecutar pruebas
            results = []
            all_passed = True
            
            for test_file in test_files:
                logger.info(f"Ejecutando pruebas en {test_file}")
                
                # Ejecutar prueba en proceso separado
                result = subprocess.run(
                    [sys.executable, str(test_file)],
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root)
                )
                
                test_passed = result.returncode == 0
                all_passed = all_passed and test_passed
                
                results.append({
                    "file": str(test_file),
                    "passed": test_passed,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                })
            
            return {
                "status": "success",
                "passed": all_passed,
                "test_count": len(test_files),
                "results": results,
                "execution_time": time.time() - start_time
            }
        
        except Exception as e:
            logger.error(f"Error al ejecutar pruebas: {e}")
            return {
                "status": "error",
                "error": str(e),
                "passed": False
            }
    
    def _validate_code_changes(self, original_code: str, modified_code: str) -> Dict[str, Any]:
        """
        Valida cambios en el código para asegurar que son seguros.
        
        Args:
            original_code: Código original.
            modified_code: Código modificado.
            
        Returns:
            Diccionario con resultado de validación.
        """
        try:
            # Verificar que el código modificado es sintácticamente válido
            try:
                ast.parse(modified_code)
            except SyntaxError as e:
                return {
                    "valid": False,
                    "error": f"Error de sintaxis: {str(e)}",
                    "line": e.lineno,
                    "offset": e.offset
                }
            
            # Analizar cambios estructurales
            original_tree = ast.parse(original_code)
            modified_tree = ast.parse(modified_code)
            
            # Extraer clases y funciones originales
            original_classes = {node.name: node for node in ast.walk(original_tree) if isinstance(node, ast.ClassDef)}
            original_functions = {node.name: node for node in ast.walk(original_tree) if isinstance(node, ast.FunctionDef)}
            
            # Extraer clases y funciones modificadas
            modified_classes = {node.name: node for node in ast.walk(modified_tree) if isinstance(node, ast.ClassDef)}
            modified_functions = {node.name: node for node in ast.walk(modified_tree) if isinstance(node, ast.FunctionDef)}
            
            # Verificar eliminación de clases o funciones
            removed_classes = set(original_classes.keys()) - set(modified_classes.keys())
            removed_functions = set(original_functions.keys()) - set(modified_functions.keys())
            
            if removed_classes and not self.safety_checks["allow_structural_changes"]:
                return {
                    "valid": False,
                    "error": f"Eliminación de clases no permitida: {', '.join(removed_classes)}",
                    "structural_changes": True
                }
            
            if removed_functions and not self.safety_checks["allow_structural_changes"]:
                return {
                    "valid": False,
                    "error": f"Eliminación de funciones no permitida: {', '.join(removed_functions)}",
                    "structural_changes": True
                }
            
            # Verificar cambios en firmas de métodos
            for func_name, orig_func in original_functions.items():
                if func_name in modified_functions:
                    mod_func = modified_functions[func_name]
                    
                    # Comparar argumentos
                    orig_args = [arg.arg for arg in orig_func.args.args]
                    mod_args = [arg.arg for arg in mod_func.args.args]
                    
                    if orig_args != mod_args and not self.safety_checks["allow_structural_changes"]:
                        return {
                            "valid": False,
                            "error": f"Cambio en firma de función no permitido: {func_name}",
                            "structural_changes": True
                        }
            
            # Verificar importaciones sospechosas
            suspicious_imports = ["os.system", "subprocess", "eval", "exec", "__import__"]
            
            for node in ast.walk(modified_tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name in ["os", "subprocess", "sys"]:
                            # Verificar si estas importaciones ya existían
                            existed = False
                            for orig_node in ast.walk(
(Content truncated due to size limit. Use line ranges to read in chunks)