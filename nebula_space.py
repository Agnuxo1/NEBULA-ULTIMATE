"""
Implementación de la clase NebulaSpace para NEBULA.

Esta clase gestiona el entorno espacial dinámico donde existen y se conectan
las neuronas cuánticas, clusters y sectores.
"""

import logging
import time
import random
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from collections import defaultdict

import numpy as np
import torch
import networkx as nx
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from core.quantum_neuron import QuantumNeuron
from core.cluster import Cluster, Sector
from utils.config import PARAMETERS
from utils.helpers import convert_to_numpy

logger = logging.getLogger("NEBULA.NebulaSpace")

class NebulaSpace:
    """
    Gestiona el entorno espacial dinámico para las neuronas, clusters y sectores de NEBULA.
    
    NebulaSpace proporciona un entorno tridimensional donde las neuronas pueden:
    - Existir en posiciones específicas
    - Conectarse entre sí
    - Emitir y recibir señales (luz)
    - Organizarse en clusters y sectores
    - Evolucionar y adaptarse con el tiempo
    """
    
    def __init__(
        self,
        dimensions: Tuple[int, int, int] = PARAMETERS["SPACE_DIMENSIONS"],
        device: torch.device = PARAMETERS["DEVICE"]
    ):
        """
        Inicializa el espacio de NEBULA.
        
        Args:
            dimensions: Dimensiones del espacio 3D (x, y, z).
            device: Dispositivo de PyTorch para cálculos tensoriales.
        """
        logger.info(f"Inicializando NebulaSpace en {device} con dimensiones {dimensions}")
        self.dimensions = torch.tensor(dimensions, dtype=torch.float32, device=device)
        self.device = device
        
        # Diccionarios para búsqueda eficiente por ID
        self.neurons: Dict[int, QuantumNeuron] = {}
        self.clusters: Dict[str, Cluster] = {}
        self.sectors: Dict[str, Sector] = {}
        
        # Contadores para generación de IDs
        self.next_neuron_id = 0
        self.next_cluster_id = 0
        self.next_sector_id = 0
        
        # Estructura para búsqueda espacial eficiente
        self.kdtree = None
        self.kdtree_needs_update = True
        
        # Grafo de conexiones entre neuronas
        self.connection_graph = nx.Graph()
    
    def _get_new_id(self, type_name: str) -> Union[int, str]:
        """
        Genera un ID único para neuronas, clusters o sectores.
        
        Args:
            type_name: Tipo de entidad ('neuron', 'cluster', o 'sector').
            
        Returns:
            ID único para la entidad.
            
        Raises:
            ValueError: Si el tipo de entidad es desconocido.
        """
        if type_name == "neuron":
            self.next_neuron_id += 1
            return self.next_neuron_id - 1
        elif type_name == "cluster":
            self.next_cluster_id += 1
            return f"C{self.next_cluster_id - 1}"
        elif type_name == "sector":
            self.next_sector_id += 1
            return f"S{self.next_sector_id - 1}"
        else:
            raise ValueError(f"Tipo de ID desconocido: {type_name}")
    
    def add_neuron(self, neuron: QuantumNeuron, assign_structure: bool = True):
        """
        Añade una neurona al espacio.
        
        Args:
            neuron: Neurona a añadir.
            assign_structure: Si es True, asigna la neurona a un cluster/sector.
        """
        if neuron.id in self.neurons:
            logger.warning(f"Neurona {neuron.id} ya existe. Omitiendo adición.")
            return
        
        if not isinstance(neuron, QuantumNeuron):
            logger.error(f"Se intentó añadir un objeto que no es QuantumNeuron con ID {neuron.id}")
            return
        
        self.neurons[neuron.id] = neuron
        self.connection_graph.add_node(neuron.id)
        self.kdtree_needs_update = True
        
        if assign_structure:
            self._assign_neuron_to_structure(neuron)
    
    def remove_neuron(self, neuron_id: int):
        """
        Elimina una neurona del espacio y limpia referencias.
        
        Args:
            neuron_id: ID de la neurona a eliminar.
        """
        if neuron_id not in self.neurons:
            logger.warning(f"Neurona {neuron_id} no encontrada para eliminación.")
            return
        
        neuron = self.neurons.pop(neuron_id)
        logger.debug(f"Eliminando neurona {neuron_id}")
        
        # Eliminar del grafo
        if self.connection_graph.has_node(neuron_id):
            edges_to_remove = list(self.connection_graph.edges(neuron_id))
            self.connection_graph.remove_edges_from(edges_to_remove)
            self.connection_graph.remove_node(neuron_id)
        
        # Actualizar listas de conexiones de otras neuronas
        for other_neuron in self.neurons.values():
            other_neuron.connections = [(target_id, strength) for target_id, strength 
                                       in other_neuron.connections if target_id != neuron_id]
        
        # Eliminar del Cluster
        if neuron.cluster_id and neuron.cluster_id in self.clusters:
            cluster = self.clusters[neuron.cluster_id]
            cluster.remove_neuron(neuron)
            if not cluster.neuron_ids:  # Si el cluster queda vacío
                self._remove_empty_cluster(cluster.id)
        
        # Limpiar referencia al sector
        neuron.sector_id = None
        
        self.kdtree_needs_update = True
    
    def _remove_empty_cluster(self, cluster_id: str):
        """
        Elimina un cluster vacío y potencialmente su sector padre si queda vacío.
        
        Args:
            cluster_id: ID del cluster a eliminar.
        """
        if cluster_id not in self.clusters:
            return
        
        cluster = self.clusters.pop(cluster_id)
        logger.info(f"Eliminando cluster vacío {cluster_id}")
        
        # Encontrar el sector que contiene este cluster y eliminar referencia
        found_sector = None
        for sector in self.sectors.values():
            if cluster_id in sector.cluster_ids:
                sector.remove_cluster(cluster)
                found_sector = sector
                break
        
        if found_sector and not found_sector.cluster_ids:
            self._remove_empty_sector(found_sector.id)
    
    def _remove_empty_sector(self, sector_id: str):
        """
        Elimina un sector vacío.
        
        Args:
            sector_id: ID del sector a eliminar.
        """
        if sector_id in self.sectors:
            del self.sectors[sector_id]
            logger.info(f"Eliminando sector vacío {sector_id}")
    
    def _assign_neuron_to_structure(self, neuron: QuantumNeuron):
        """
        Asigna una neurona al cluster/sector más cercano, creando nuevos si es necesario.
        
        Args:
            neuron: Neurona a asignar.
        """
        if not self.sectors:  # Crear primer sector si no existe ninguno
            sector_id = self._get_new_id("sector")
            # Colocar sector cerca de la neurona
            sector_pos = neuron.position.data + (torch.rand_like(neuron.position.data) - 0.5) * 10
            sector = Sector(sector_id, sector_pos.clamp(0, self.dimensions - 1), self.device)
            self.sectors[sector_id] = sector
            
            # Crear primer cluster en este sector
            cluster_id = self._get_new_id("cluster")
            cluster_pos = neuron.position.data + (torch.rand_like(neuron.position.data) - 0.5) * 5
            cluster = Cluster(cluster_id, cluster_pos.clamp(0, self.dimensions - 1), self.device)
            self.clusters[cluster_id] = cluster
            
            # Añadir cluster al sector
            sector.add_cluster(cluster)
            
            # Añadir neurona al cluster
            cluster.add_neuron(neuron)
            neuron.sector_id = sector_id
            
            return
        
        # Si ya existen clusters, encontrar el más cercano
        if self.clusters:
            # Calcular distancias a todos los clusters
            cluster_positions = torch.stack([c.position for c in self.clusters.values()])
            distances = torch.norm(cluster_positions - neuron.position.data, dim=1)
            
            # Encontrar el cluster más cercano
            min_idx = torch.argmin(distances)
            closest_cluster_id = list(self.clusters.keys())[min_idx.item()]
            closest_cluster = self.clusters[closest_cluster_id]
            
            # Si la distancia es razonable, asignar a ese cluster
            if distances[min_idx] < 20.0:  # Umbral de distancia ajustable
                closest_cluster.add_neuron(neuron)
                # Asignar al sector del cluster
                for sector in self.sectors.values():
                    if closest_cluster.id in sector.cluster_ids:
                        neuron.sector_id = sector.id
                        break
                return
        
        # Si no hay clusters cercanos, crear uno nuevo en el sector más cercano
        sector_positions = torch.stack([s.position for s in self.sectors.values()])
        distances = torch.norm(sector_positions - neuron.position.data, dim=1)
        min_idx = torch.argmin(distances)
        closest_sector_id = list(self.sectors.keys())[min_idx.item()]
        closest_sector = self.sectors[closest_sector_id]
        
        # Crear nuevo cluster
        cluster_id = self._get_new_id("cluster")
        cluster_pos = neuron.position.data + (torch.rand_like(neuron.position.data) - 0.5) * 3
        cluster = Cluster(cluster_id, cluster_pos.clamp(0, self.dimensions - 1), self.device)
        self.clusters[cluster_id] = cluster
        
        # Añadir cluster al sector
        closest_sector.add_cluster(cluster)
        
        # Añadir neurona al cluster
        cluster.add_neuron(neuron)
        neuron.sector_id = closest_sector_id
    
    def initialize_neurons(self, num_neurons: int):
        """
        Inicializa un número específico de neuronas en el espacio.
        
        Args:
            num_neurons: Número de neuronas a crear.
        """
        logger.info(f"Inicializando {num_neurons} neuronas en el espacio...")
        
        added_count = 0
        for _ in range(num_neurons):
            try:
                # Generar posición aleatoria dentro de las dimensiones del espacio
                position = torch.rand(3, device=self.device) * self.dimensions
                
                # Crear neurona con ID único
                neuron_id = self._get_new_id("neuron")
                neuron = QuantumNeuron(
                    neuron_id=neuron_id,
                    position=position,
                    input_dim=PARAMETERS["INPUT_DIM"],
                    num_qubits=PARAMETERS["NUM_QUBITS"],
                    num_layers=PARAMETERS["QUANTUM_LAYERS"],
                    device=self.device
                )
                
                # Añadir neurona al espacio sin asignar estructura todavía
                self.add_neuron(neuron, assign_structure=False)
                added_count += 1
            except Exception as e:
                logger.error(f"Error al inicializar neurona {neuron_id}: {e}")
                # Revertir contador de ID si falla la inicialización
                self.next_neuron_id -= 1
        
        logger.info(f"Añadidas con éxito {added_count}/{num_neurons} neuronas.")
        
        # Asignar estructura después de añadir todas las neuronas
        logger.info("Asignando estructura inicial...")
        if self.neurons:
            # La asignación inicial puede ser simple o usar update_structure
            self.update_structure()  # Usar K-Means para estructura inicial
        
        self.kdtree_needs_update = True
    
    def _update_kdtree(self):
        """Actualiza el KDTree para búsqueda espacial eficiente."""
        if not self.neurons:
            self.kdtree = None
            return
        
        positions = np.array([convert_to_numpy(n.position.data) for n in self.neurons.values()])
        ids = list(self.neurons.keys())
        
        self.kdtree = cKDTree(positions)
        self.kdtree_data = {
            'positions': positions,
            'ids': ids
        }
        self.kdtree_needs_update = False
    
    def find_neurons_within_radius(self, position: torch.Tensor, radius: float) -> List[QuantumNeuron]:
        """
        Encuentra neuronas dentro de un radio específico de una posición.
        
        Args:
            position: Posición central para la búsqueda.
            radius: Radio de búsqueda.
            
        Returns:
            Lista de neuronas dentro del radio.
        """
        if self.kdtree_needs_update:
            self._update_kdtree()
        
        if self.kdtree is None:
            return []
        
        pos_np = convert_to_numpy(position)
        indices = self.kdtree.query_ball_point(pos_np, radius)
        
        if not indices:
            return []
        
        # Convertir índices a IDs de neurona y luego a objetos neurona
        neuron_ids = [self.kdtree_data['ids'][i] for i in indices]
        return [self.neurons[nid] for nid in neuron_ids if nid in self.neurons]
    
    def calculate_light_intensity(self, emitter: QuantumNeuron, receiver_position: torch.Tensor) -> float:
        """
        Calcula la intensidad de luz que llega de un emisor a una posición receptora.
        
        Args:
            emitter: Neurona emisora.
            receiver_position: Posición del receptor.
            
        Returns:
            Intensidad de luz en la posición receptora.
        """
        # Modelo simple: intensidad disminuye con el cuadrado de la distancia
        distance_sq = torch.sum((emitter.position.data - receiver_position) ** 2).item()
        base_intensity = emitter.emit_light()
        
        # Evitar división por cero
        if distance_sq < 0.0001:
            distance_sq = 0.0001
        
        # Factor de atenuación
        attenuation_factor = PARAMETERS.get("LIGHT_ATTENUATION_FACTOR", 0.3)
        
        # Intensidad = luminosidad / (distancia^2 * factor)
        intensity = base_intensity / (distance_sq * attenuation_factor)
        
        # Limitar intensidad máxima
        return min(intensity, 1.0)
    
    def run_simulation_step(self, iteration: int):
        """
        Ejecuta un paso de simulación en el espacio de NEBULA.
        
        Args:
            iteration: Número de iteración actual.
        """
        start_time = time.time()
        logger.debug(f"Ejecutando paso de simulación {iteration}...")
        
        # 1. Propagación de Luz
        self._propagate_light_step()
        
        # 2. Actualización de Conexiones
        self._update_connections_step()
        
        # 3. Actualización de Posiciones
        self._update_positions_step()
        
        # 4. Decaimiento de Luminosidad y Poda de Neuronas Inactivas
        self._decay_and_prune_step(iteration)
        
        # 5. Actualización de Estructura (menos frecuente)
        if iteration % PARAMETERS.get("STRUCTURE_UPDATE_INTERVAL", 25) == 0:
            self.update_structure()
        
        # Marcar KDTree para actualización
        self.kdtree_needs_update = True
        
        elapsed = time.time() - start_time
        logger.debug(f"Paso de simulación {iteration} completado en {elapsed:.4f} segundos")
    
    def _propagate_light_step(self):
        """Calcula y aplica la intensidad de luz recibida por cada neurona."""
        num_neurons = len(self.neurons)
        if num_neurons < 2:
            return
        
        # Almacenar intensidades para aplicar actualizaciones simultáneamente
        received_intensities = defaultdict(float)
        all_neurons_list = list(self.neurons.values())
        
        # Iteración N^2 simple (puede optimizarse con KDTree para interacciones dispersas)
(Content truncated due to size limit. Use line ranges to read in chunks)