"""
Implementación de la clase QuantumNeuron para NEBULA.

Esta clase representa una neurona cuántica que combina un circuito cuántico simulado
con propiedades clásicas como posición y luminosidad.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as pnp

from utils.config import PARAMETERS
from utils.helpers import convert_to_numpy

logger = logging.getLogger("NEBULA.QuantumNeuron")

class QuantumNeuron(nn.Module):
    """
    Neurona cuántica que combina un circuito cuántico simulado con propiedades clásicas.
    
    Cada neurona tiene:
    - Un circuito cuántico simulado mediante PennyLane
    - Una posición en el espacio 3D
    - Una luminosidad que representa su nivel de activación
    - Conexiones con otras neuronas
    """
    
    def __init__(
        self,
        neuron_id: Optional[int] = None,
        position: Union[np.ndarray, torch.Tensor, List[float]] = None,
        input_dim: int = PARAMETERS["INPUT_DIM"],
        num_qubits: int = PARAMETERS["NUM_QUBITS"],
        num_layers: int = PARAMETERS["QUANTUM_LAYERS"],
        device: torch.device = PARAMETERS["DEVICE"]
    ):
        """
        Inicializa una neurona cuántica.
        
        Args:
            neuron_id: Identificador único de la neurona. Si es None, se genera automáticamente.
            position: Posición 3D de la neurona en el espacio. Si es None, se genera aleatoriamente.
            input_dim: Dimensión de la entrada clásica que afecta al estado cuántico.
            num_qubits: Número de qubits en el circuito cuántico.
            num_layers: Número de capas en el circuito cuántico.
            device: Dispositivo de PyTorch para cálculos tensoriales.
        """
        super().__init__()
        
        # Identificador único
        self.id = neuron_id if neuron_id is not None else uuid.uuid4().int % 10000000
        
        # Configuración del dispositivo
        self._device = device
        
        # Dimensiones del circuito cuántico
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        
        # Inicialización del circuito cuántico
        self.quantum_circuit = None
        self.qlayer_weights = nn.Parameter(
            torch.randn(num_layers, num_qubits, 3, device=self._device, requires_grad=True)
        )
        
        # Inicialización del dispositivo cuántico
        try:
            # Usar simulador de PennyLane
            self.qdev = qml.device("default.qubit", wires=self.num_qubits)
            
            # Definir el circuito cuántico
            @qml.qnode(self.qdev, interface="torch")
            def quantum_circuit_node(inputs, weights):
                # Codificar entradas clásicas en el estado cuántico
                for i in range(self.num_qubits):
                    qml.RY(inputs[i % len(inputs)], wires=i)
                
                # Aplicar capas parametrizadas
                qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
                
                # Medir expectativas de Pauli-Z para cada qubit
                return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
            
            self.quantum_circuit = quantum_circuit_node
        except Exception as e:
            logger.warning(f"Neurona {self.id}: Error al inicializar circuito cuántico: {e}")
            logger.warning(f"Neurona {self.id}: Circuito cuántico deshabilitado, usando fallback clásico.")
        
        # Propiedades clásicas
        # Posición en el espacio 3D
        if position is None:
            # Generar posición aleatoria
            position = torch.rand(3, device=self._device)
        elif not isinstance(position, torch.Tensor):
            # Convertir a tensor de PyTorch
            pos_np = convert_to_numpy(position)
            if pos_np is None:
                raise ValueError("Tipo de posición inválido para Neurona")
            position = torch.tensor(pos_np, dtype=torch.float32, device=self._device)
        else:
            position = position.to(dtype=torch.float32, device=self._device)
        
        self.position = nn.Parameter(position)
        
        # Luminosidad (nivel de activación)
        self.luminosity = nn.Parameter(torch.rand(1, device=self._device) * 0.3 + 0.1)
        
        # Conexiones con otras neuronas (id_neurona_destino, fuerza)
        self.connections: List[Tuple[int, float]] = []
        
        # Metadatos
        self.last_activity_time = time.time()
        self.cluster_id: Optional[str] = None
        self.sector_id: Optional[str] = None
    
    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Pasa la entrada a través del circuito cuántico y combina con el estado clásico.
        
        Args:
            x: Entrada clásica que afecta al estado cuántico.
            
        Returns:
            Tensor que representa la salida/estado de la neurona.
        """
        # Procesamiento de entrada
        if isinstance(x, np.ndarray):
            input_tensor = torch.from_numpy(x).float().to(self._device)
        elif isinstance(x, torch.Tensor):
            input_tensor = x.float().to(self._device)
        else:
            logger.error(f"Neurona {self.id}: Tipo de entrada inválido {type(x)}.")
            return torch.zeros(self.num_qubits, device=self._device)
        
        # Ajustar dimensiones de entrada
        input_tensor = input_tensor.flatten()
        current_dim = input_tensor.shape[0]
        
        if current_dim > self.input_dim:
            input_tensor = input_tensor[:self.input_dim]
        elif current_dim < self.input_dim:
            padding = torch.zeros(self.input_dim - current_dim, device=self._device)
            input_tensor = torch.cat((input_tensor, padding))
        
        # Procesamiento cuántico
        if self.quantum_circuit:
            try:
                q_output = self.quantum_circuit(input_tensor, self.qlayer_weights)
                quantum_state = torch.stack(q_output)
            except Exception as e:
                logger.error(f"Neurona {self.id}: Error durante ejecución del circuito cuántico: {e}")
                quantum_state = torch.zeros(self.num_qubits, device=self._device)
        else:
            # Fallback clásico si el circuito cuántico está deshabilitado
            quantum_state = torch.zeros(self.num_qubits, device=self._device)
        
        # Combinar estado cuántico y clásico
        final_output = quantum_state * self.luminosity.clamp(0.0, 1.0)
        
        self.last_activity_time = time.time()
        return final_output
    
    def emit_light(self) -> float:
        """
        Calcula la intensidad de luz emitida por la neurona.
        
        Returns:
            Intensidad de luz emitida.
        """
        return self.luminosity.item()
    
    def receive_light(self, intensity: float):
        """
        Procesa la luz recibida, actualizando el estado interno.
        
        Args:
            intensity: Intensidad de luz recibida.
        """
        update_factor = PARAMETERS.get("LIGHT_RECEIVE_FACTOR", 0.05)
        self.luminosity.data += intensity * update_factor
        self.luminosity.data.clamp_(0.0, 1.0)
        self.last_activity_time = time.time()
    
    def decay_luminosity(self):
        """Aplica un factor de decaimiento a la luminosidad."""
        decay_rate = PARAMETERS.get("LUMINOSITY_DECAY", 0.01)
        self.luminosity.data *= (1.0 - decay_rate)
        self.luminosity.data.clamp_(0.0, 1.0)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Devuelve el estado serializable de la neurona.
        
        Returns:
            Diccionario con el estado de la neurona.
        """
        state = {
            "id": self.id,
            "position": convert_to_numpy(self.position.data),
            "luminosity": self.luminosity.item(),
            "qlayer_weights": convert_to_numpy(self.qlayer_weights.data),
            "connections": self.connections.copy(),
            "last_activity_time": self.last_activity_time,
            "input_dim": self.input_dim,
            "num_qubits": self.num_qubits,
            "num_layers": self.num_layers,
            "cluster_id": self.cluster_id,
            "sector_id": self.sector_id,
        }
        return state
    
    @classmethod
    def from_state(cls, state: Dict[str, Any], device: torch.device) -> "QuantumNeuron":
        """
        Crea una instancia de neurona a partir de un estado guardado.
        
        Args:
            state: Estado guardado de la neurona.
            device: Dispositivo de PyTorch.
            
        Returns:
            Instancia de QuantumNeuron.
        """
        neuron = cls(
            neuron_id=state["id"],
            position=state["position"],
            input_dim=state.get("input_dim", PARAMETERS["INPUT_DIM"]),
            num_qubits=state.get("num_qubits", PARAMETERS["NUM_QUBITS"]),
            num_layers=state.get("num_layers", PARAMETERS["QUANTUM_LAYERS"]),
            device=device
        )
        
        # Cargar parámetros
        neuron.luminosity.data = torch.tensor([state["luminosity"]], device=device)
        if 'qlayer_weights' in state and state['qlayer_weights'] is not None:
            neuron.qlayer_weights.data = torch.tensor(state["qlayer_weights"], dtype=torch.float32, device=device)
        
        # Asegurar que la posición se carga correctamente
        neuron.position.data = torch.tensor(state["position"], dtype=torch.float32, device=device)
        
        neuron.connections = state.get("connections", [])
        neuron.last_activity_time = state.get("last_activity_time", time.time())
        neuron.cluster_id = state.get("cluster_id")
        neuron.sector_id = state.get("sector_id")
        
        return neuron
    
    def get_embedding(self) -> np.ndarray:
        """
        Obtiene un vector de características que representa el estado de la neurona para clustering.
        
        Returns:
            Vector de características.
        """
        pos = convert_to_numpy(self.position.data)
        lum = self.luminosity.item()
        weights_agg = np.linalg.norm(convert_to_numpy(self.qlayer_weights.data)) if self.qlayer_weights is not None else 0.0
        num_conn = len(self.connections)
        
        embedding = np.array([pos[0], pos[1], pos[2], lum, weights_agg, num_conn], dtype=np.float32)
        return embedding
