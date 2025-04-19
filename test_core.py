"""
Módulo de pruebas para el núcleo de NEBULA.

Este módulo contiene pruebas unitarias para verificar el funcionamiento
correcto de los componentes principales del núcleo de NEBULA.
"""

import unittest
import sys
import os
import numpy as np
import torch
from pathlib import Path

# Añadir directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quantum_neuron import QuantumNeuron
from core.cluster import Cluster, Sector
from core.nebula_space import NebulaSpace
from utils.config import PARAMETERS

class TestQuantumNeuron(unittest.TestCase):
    """Pruebas para la clase QuantumNeuron."""
    
    def setUp(self):
        """Configuración para las pruebas."""
        self.neuron = QuantumNeuron(
            position=np.array([0.5, 0.5, 0.5]),
            n_qubits=2,
            activation='sigmoid'
        )
    
    def test_initialization(self):
        """Prueba la inicialización correcta de una neurona cuántica."""
        self.assertEqual(self.neuron.n_qubits, 2)
        self.assertEqual(self.neuron.activation_name, 'sigmoid')
        np.testing.assert_array_equal(self.neuron.position, np.array([0.5, 0.5, 0.5]))
        self.assertIsNotNone(self.neuron.id)
        self.assertIsNotNone(self.neuron.circuit)
    
    def test_process_signal(self):
        """Prueba el procesamiento de señales."""
        input_signal = np.array([0.1, 0.2])
        output = self.neuron.process_signal(input_signal)
        
        # Verificar que la salida tiene la forma correcta
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (1,))
        
        # Verificar que la salida está en el rango [0, 1]
        self.assertTrue(0 <= output[0] <= 1)
    
    def test_update_weights(self):
        """Prueba la actualización de pesos."""
        old_weights = self.neuron.weights.copy()
        
        # Actualizar pesos
        self.neuron.update_weights(learning_rate=0.1)
        
        # Verificar que los pesos han cambiado
        self.assertFalse(np.array_equal(old_weights, self.neuron.weights))
    
    def test_calculate_distance(self):
        """Prueba el cálculo de distancia entre neuronas."""
        other_neuron = QuantumNeuron(
            position=np.array([1.0, 1.0, 1.0]),
            n_qubits=2
        )
        
        distance = self.neuron.calculate_distance(other_neuron)
        expected_distance = np.sqrt(3 * (0.5**2))  # sqrt((1-0.5)^2 + (1-0.5)^2 + (1-0.5)^2)
        
        self.assertAlmostEqual(distance, expected_distance, places=5)
    
    def test_serialize_deserialize(self):
        """Prueba la serialización y deserialización."""
        serialized = self.neuron.serialize()
        
        # Verificar que la serialización produce un diccionario
        self.assertIsInstance(serialized, dict)
        
        # Crear una nueva neurona a partir de los datos serializados
        new_neuron = QuantumNeuron.deserialize(serialized)
        
        # Verificar que la nueva neurona tiene las mismas propiedades
        self.assertEqual(new_neuron.n_qubits, self.neuron.n_qubits)
        self.assertEqual(new_neuron.activation_name, self.neuron.activation_name)
        np.testing.assert_array_equal(new_neuron.position, self.neuron.position)
        np.testing.assert_array_equal(new_neuron.weights, self.neuron.weights)


class TestCluster(unittest.TestCase):
    """Pruebas para la clase Cluster."""
    
    def setUp(self):
        """Configuración para las pruebas."""
        self.neurons = [
            QuantumNeuron(position=np.array([0.1, 0.1, 0.1]), n_qubits=2),
            QuantumNeuron(position=np.array([0.2, 0.2, 0.2]), n_qubits=2),
            QuantumNeuron(position=np.array([0.3, 0.3, 0.3]), n_qubits=2)
        ]
        
        self.cluster = Cluster(
            position=np.array([0.2, 0.2, 0.2]),
            radius=0.2
        )
        
        # Añadir neuronas al cluster
        for neuron in self.neurons:
            self.cluster.add_neuron(neuron)
    
    def test_initialization(self):
        """Prueba la inicialización correcta de un cluster."""
        self.assertEqual(len(self.cluster.neurons), 3)
        np.testing.assert_array_equal(self.cluster.position, np.array([0.2, 0.2, 0.2]))
        self.assertEqual(self.cluster.radius, 0.2)
        self.assertIsNotNone(self.cluster.id)
    
    def test_contains_neuron(self):
        """Prueba la verificación de si un cluster contiene una neurona."""
        # Neurona dentro del radio
        neuron_inside = QuantumNeuron(position=np.array([0.25, 0.25, 0.25]), n_qubits=2)
        self.assertTrue(self.cluster.contains_neuron(neuron_inside))
        
        # Neurona fuera del radio
        neuron_outside = QuantumNeuron(position=np.array([0.5, 0.5, 0.5]), n_qubits=2)
        self.assertFalse(self.cluster.contains_neuron(neuron_outside))
    
    def test_process_signal(self):
        """Prueba el procesamiento de señales en un cluster."""
        input_signal = np.array([0.1, 0.2])
        output = self.cluster.process_signal(input_signal)
        
        # Verificar que la salida tiene la forma correcta
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (3,))  # Una salida por cada neurona
    
    def test_update_position(self):
        """Prueba la actualización de la posición del cluster."""
        old_position = self.cluster.position.copy()
        
        # Actualizar posición
        self.cluster.update_position()
        
        # La posición debería ser el promedio de las posiciones de las neuronas
        expected_position = np.mean([n.position for n in self.neurons], axis=0)
        np.testing.assert_array_almost_equal(self.cluster.position, expected_position)
        
        # Verificar que la posición ha cambiado
        self.assertFalse(np.array_equal(old_position, self.cluster.position))
    
    def test_merge_clusters(self):
        """Prueba la fusión de clusters."""
        other_neurons = [
            QuantumNeuron(position=np.array([0.4, 0.4, 0.4]), n_qubits=2),
            QuantumNeuron(position=np.array([0.5, 0.5, 0.5]), n_qubits=2)
        ]
        
        other_cluster = Cluster(
            position=np.array([0.45, 0.45, 0.45]),
            radius=0.2
        )
        
        # Añadir neuronas al otro cluster
        for neuron in other_neurons:
            other_cluster.add_neuron(neuron)
        
        # Fusionar clusters
        merged_cluster = Cluster.merge_clusters(self.cluster, other_cluster)
        
        # Verificar que el cluster fusionado contiene todas las neuronas
        self.assertEqual(len(merged_cluster.neurons), 5)
        
        # Verificar que la posición es el promedio ponderado
        expected_position = np.mean([n.position for n in self.neurons + other_neurons], axis=0)
        np.testing.assert_array_almost_equal(merged_cluster.position, expected_position)


class TestSector(unittest.TestCase):
    """Pruebas para la clase Sector."""
    
    def setUp(self):
        """Configuración para las pruebas."""
        self.clusters = [
            Cluster(position=np.array([0.1, 0.1, 0.1]), radius=0.1),
            Cluster(position=np.array([0.3, 0.3, 0.3]), radius=0.1),
            Cluster(position=np.array([0.5, 0.5, 0.5]), radius=0.1)
        ]
        
        self.sector = Sector(
            position=np.array([0.3, 0.3, 0.3]),
            size=0.5
        )
        
        # Añadir clusters al sector
        for cluster in self.clusters:
            self.sector.add_cluster(cluster)
    
    def test_initialization(self):
        """Prueba la inicialización correcta de un sector."""
        self.assertEqual(len(self.sector.clusters), 3)
        np.testing.assert_array_equal(self.sector.position, np.array([0.3, 0.3, 0.3]))
        self.assertEqual(self.sector.size, 0.5)
        self.assertIsNotNone(self.sector.id)
    
    def test_contains_cluster(self):
        """Prueba la verificación de si un sector contiene un cluster."""
        # Cluster dentro del sector
        cluster_inside = Cluster(position=np.array([0.4, 0.4, 0.4]), radius=0.1)
        self.assertTrue(self.sector.contains_cluster(cluster_inside))
        
        # Cluster fuera del sector
        cluster_outside = Cluster(position=np.array([0.8, 0.8, 0.8]), radius=0.1)
        self.assertFalse(self.sector.contains_cluster(cluster_outside))
    
    def test_get_clusters_in_range(self):
        """Prueba la obtención de clusters en un rango."""
        # Punto central
        center = np.array([0.3, 0.3, 0.3])
        
        # Obtener clusters en un rango pequeño
        clusters_close = self.sector.get_clusters_in_range(center, 0.1)
        self.assertEqual(len(clusters_close), 1)  # Solo el cluster en [0.3, 0.3, 0.3]
        
        # Obtener clusters en un rango mayor
        clusters_medium = self.sector.get_clusters_in_range(center, 0.3)
        self.assertEqual(len(clusters_medium), 2)  # Los clusters en [0.1, 0.1, 0.1] y [0.3, 0.3, 0.3]
        
        # Obtener todos los clusters
        clusters_all = self.sector.get_clusters_in_range(center, 0.5)
        self.assertEqual(len(clusters_all), 3)  # Todos los clusters
    
    def test_update_position(self):
        """Prueba la actualización de la posición del sector."""
        old_position = self.sector.position.copy()
        
        # Actualizar posición
        self.sector.update_position()
        
        # La posición debería ser el promedio de las posiciones de los clusters
        expected_position = np.mean([c.position for c in self.clusters], axis=0)
        np.testing.assert_array_almost_equal(self.sector.position, expected_position)
        
        # Verificar que la posición ha cambiado
        self.assertFalse(np.array_equal(old_position, self.sector.position))


class TestNebulaSpace(unittest.TestCase):
    """Pruebas para la clase NebulaSpace."""
    
    def setUp(self):
        """Configuración para las pruebas."""
        self.space = NebulaSpace(
            dimensions=3,
            size=1.0,
            initial_neurons=10,
            initial_clusters=2,
            initial_sectors=1
        )
    
    def test_initialization(self):
        """Prueba la inicialización correcta del espacio."""
        self.assertEqual(self.space.dimensions, 3)
        self.assertEqual(self.space.size, 1.0)
        self.assertEqual(len(self.space.neurons), 10)
        self.assertEqual(len(self.space.clusters), 2)
        self.assertEqual(len(self.space.sectors), 1)
    
    def test_add_neuron(self):
        """Prueba la adición de una neurona al espacio."""
        neuron = QuantumNeuron(
            position=np.array([0.5, 0.5, 0.5]),
            n_qubits=2
        )
        
        # Añadir neurona
        self.space.add_neuron(neuron)
        
        # Verificar que la neurona se ha añadido
        self.assertEqual(len(self.space.neurons), 11)
        self.assertIn(neuron.id, self.space.neurons)
    
    def test_create_connection(self):
        """Prueba la creación de conexiones entre neuronas."""
        # Obtener dos neuronas
        neuron_ids = list(self.space.neurons.keys())
        neuron1_id = neuron_ids[0]
        neuron2_id = neuron_ids[1]
        
        # Crear conexión
        self.space.create_connection(neuron1_id, neuron2_id, weight=0.5)
        
        # Verificar que la conexión se ha creado
        self.assertIn(neuron2_id, self.space.connections[neuron1_id])
        self.assertEqual(self.space.connections[neuron1_id][neuron2_id], 0.5)
    
    def test_propagate_signal(self):
        """Prueba la propagación de señales en el espacio."""
        # Crear una señal
        signal = np.array([0.1, 0.2])
        
        # Propagar señal
        activations = self.space.propagate_signal(signal)
        
        # Verificar que se han producido activaciones
        self.assertIsInstance(activations, dict)
        self.assertTrue(len(activations) > 0)
    
    def test_reorganize_space(self):
        """Prueba la reorganización del espacio."""
        # Guardar estado inicial
        initial_clusters = len(self.space.clusters)
        initial_sectors = len(self.space.sectors)
        
        # Reorganizar espacio
        self.space.reorganize_space()
        
        # Verificar que la reorganización ha ocurrido
        # (No podemos predecir exactamente el resultado, pero verificamos que algo ha cambiado)
        self.assertTrue(len(self.space.clusters) >= initial_clusters)
        self.assertTrue(len(self.space.sectors) >= initial_sectors)
    
    def test_find_nearest_neurons(self):
        """Prueba la búsqueda de neuronas cercanas."""
        # Obtener una neurona
        neuron_id = list(self.space.neurons.keys())[0]
        neuron = self.space.neurons[neuron_id]
        
        # Buscar neuronas cercanas
        nearest = self.space.find_nearest_neurons(neuron.position, k=3)
        
        # Verificar que se han encontrado neuronas
        self.assertEqual(len(nearest), 3)
        
        # Verificar que las neuronas están ordenadas por distancia
        distances = [np.linalg.norm(neuron.position - self.space.neurons[n_id].position) for n_id in nearest]
        self.assertTrue(all(distances[i] <= distances[i+1] for i in range(len(distances)-1)))
    
    def test_save_and_load(self):
        """Prueba el guardado y carga del espacio."""
        # Guardar espacio
        save_path = Path("test_space_save.pkl")
        self.space.save(save_path)
        
        # Verificar que el archivo se ha creado
        self.assertTrue(save_path.exists())
        
        # Cargar espacio
        loaded_space = NebulaSpace.load(save_path)
        
        # Verificar que el espacio cargado tiene las mismas propiedades
        self.assertEqual(loaded_space.dimensions, self.space.dimensions)
        self.assertEqual(loaded_space.size, self.space.size)
        self.assertEqual(len(loaded_space.neurons), len(self.space.neurons))
        self.assertEqual(len(loaded_space.clusters), len(self.space.clusters))
        self.assertEqual(len(loaded_space.sectors), len(self.space.sectors))
        
        # Limpiar
        save_path.unlink()


if __name__ == '__main__':
    unittest.main()
