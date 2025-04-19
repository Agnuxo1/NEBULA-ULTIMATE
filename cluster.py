"""
Implementación de las clases Cluster y Sector para NEBULA.

Estas clases representan agrupaciones de neuronas en el espacio de NEBULA,
permitiendo organización jerárquica y procesamiento eficiente.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Union

import torch
import numpy as np

from utils.helpers import convert_to_numpy

logger = logging.getLogger("NEBULA.Structure")

class Cluster:
    """
    Representa un cluster de neuronas en el espacio de NEBULA.
    
    Un cluster es una agrupación de neuronas que están relacionadas funcionalmente
    o espacialmente, permitiendo procesamiento más eficiente y organización jerárquica.
    """
    
    def __init__(self, cluster_id: str, position: torch.Tensor, device: torch.device):
        """
        Inicializa un cluster.
        
        Args:
            cluster_id: Identificador único del cluster.
            position: Posición central del cluster en el espacio 3D.
            device: Dispositivo de PyTorch para cálculos tensoriales.
        """
        self.id = cluster_id
        self.position = position.to(device)  # Centro del cluster
        self.neuron_ids: Set[int] = set()  # IDs de neuronas en el cluster
        self.device = device
    
    def add_neuron(self, neuron):
        """
        Añade una neurona al cluster.
        
        Args:
            neuron: Instancia de QuantumNeuron a añadir.
        """
        self.neuron_ids.add(neuron.id)
        neuron.cluster_id = self.id
    
    def remove_neuron(self, neuron):
        """
        Elimina una neurona del cluster.
        
        Args:
            neuron: Instancia de QuantumNeuron a eliminar.
        """
        self.neuron_ids.discard(neuron.id)
        if neuron.cluster_id == self.id:  # Evitar limpiar si ya fue reasignada
            neuron.cluster_id = None
    
    def update_position(self, neurons_dict: Dict[int, Any]):
        """
        Actualiza la posición del cluster basándose en el centroide de sus neuronas.
        
        Args:
            neurons_dict: Diccionario de neuronas indexado por ID.
        """
        if not self.neuron_ids:
            return
        
        positions = [neurons_dict[nid].position.data for nid in self.neuron_ids 
                    if nid in neurons_dict]
        
        if positions:
            self.position = torch.mean(torch.stack(positions), dim=0)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Devuelve el estado serializable del cluster.
        
        Returns:
            Diccionario con el estado del cluster.
        """
        return {
            "id": self.id,
            "position": convert_to_numpy(self.position),
            "neuron_ids": list(self.neuron_ids)
        }
    
    @classmethod
    def from_state(cls, state: Dict[str, Any], device: torch.device) -> "Cluster":
        """
        Crea una instancia de cluster a partir de un estado guardado.
        
        Args:
            state: Estado guardado del cluster.
            device: Dispositivo de PyTorch.
            
        Returns:
            Instancia de Cluster.
        """
        cluster = cls(
            cluster_id=state["id"],
            position=torch.tensor(state["position"], dtype=torch.float32, device=device),
            device=device
        )
        cluster.neuron_ids = set(state.get("neuron_ids", []))
        return cluster


class Sector:
    """
    Representa un sector que contiene múltiples clusters.
    
    Un sector es una agrupación de nivel superior que contiene múltiples clusters,
    permitiendo una organización jerárquica del espacio de NEBULA.
    """
    
    def __init__(self, sector_id: str, position: torch.Tensor, device: torch.device):
        """
        Inicializa un sector.
        
        Args:
            sector_id: Identificador único del sector.
            position: Posición central del sector en el espacio 3D.
            device: Dispositivo de PyTorch para cálculos tensoriales.
        """
        self.id = sector_id
        self.position = position.to(device)
        self.cluster_ids: Set[str] = set()  # IDs de clusters en el sector
        self.device = device
    
    def add_cluster(self, cluster: Cluster):
        """
        Añade un cluster al sector.
        
        Args:
            cluster: Instancia de Cluster a añadir.
        """
        self.cluster_ids.add(cluster.id)
    
    def remove_cluster(self, cluster: Cluster):
        """
        Elimina un cluster del sector.
        
        Args:
            cluster: Instancia de Cluster a eliminar.
        """
        self.cluster_ids.discard(cluster.id)
    
    def update_position(self, clusters_dict: Dict[str, Cluster]):
        """
        Actualiza la posición del sector basándose en el centroide de sus clusters.
        
        Args:
            clusters_dict: Diccionario de clusters indexado por ID.
        """
        if not self.cluster_ids:
            return
        
        positions = [clusters_dict[cid].position for cid in self.cluster_ids 
                    if cid in clusters_dict]
        
        if positions:
            self.position = torch.mean(torch.stack(positions), dim=0)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Devuelve el estado serializable del sector.
        
        Returns:
            Diccionario con el estado del sector.
        """
        return {
            "id": self.id,
            "position": convert_to_numpy(self.position),
            "cluster_ids": list(self.cluster_ids)
        }
    
    @classmethod
    def from_state(cls, state: Dict[str, Any], device: torch.device) -> "Sector":
        """
        Crea una instancia de sector a partir de un estado guardado.
        
        Args:
            state: Estado guardado del sector.
            device: Dispositivo de PyTorch.
            
        Returns:
            Instancia de Sector.
        """
        sector = cls(
            sector_id=state["id"],
            position=torch.tensor(state["position"], dtype=torch.float32, device=device),
            device=device
        )
        sector.cluster_ids = set(state.get("cluster_ids", []))
        return sector
