"""
Implementación del Procesador Óptico para NEBULA.

Esta clase simula un procesador óptico que realiza operaciones de transformación
y procesamiento de información inspiradas en sistemas ópticos.
"""

import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.config import PARAMETERS
from utils.helpers import convert_to_numpy

logger = logging.getLogger("NEBULA.OpticalProcessor")

class OpticalProcessingUnit:
    """
    Unidad de procesamiento óptico que simula operaciones de transformación
    y procesamiento de información inspiradas en sistemas ópticos.
    
    Características:
    - Transformaciones de Fourier para procesamiento en dominio de frecuencia
    - Operaciones de convolución y correlación óptica
    - Filtrado espacial y de frecuencia
    - Transformaciones holográficas
    """
    
    def __init__(self, device: torch.device = PARAMETERS["DEVICE"]):
        """
        Inicializa el procesador óptico.
        
        Args:
            device: Dispositivo de PyTorch para cálculos tensoriales.
        """
        logger.info("Inicializando OpticalProcessingUnit...")
        self.device = device
        
        # Dimensiones de procesamiento
        self.default_dim = 64
        
        # Contadores y estadísticas
        self.operations_count = 0
        self.last_operation_time = 0
        
        # Inicializar filtros predefinidos
        self.filters = self._initialize_filters()
        
        logger.info("OpticalProcessingUnit inicializada correctamente.")
    
    def _initialize_filters(self) -> Dict[str, torch.Tensor]:
        """
        Inicializa filtros predefinidos para procesamiento óptico.
        
        Returns:
            Diccionario de filtros predefinidos.
        """
        filters = {}
        
        # Filtro Gaussiano
        sigma = 2.0
        size = 7
        x = torch.arange(-(size//2), (size//2) + 1, dtype=torch.float32, device=self.device)
        y = torch.arange(-(size//2), (size//2) + 1, dtype=torch.float32, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()
        filters['gaussian'] = gaussian.unsqueeze(0).unsqueeze(0)
        
        # Filtro Laplaciano
        laplacian = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=self.device)
        filters['laplacian'] = laplacian.unsqueeze(0).unsqueeze(0)
        
        # Filtro Sobel (detección de bordes)
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32, device=self.device)
        filters['sobel_x'] = sobel_x.unsqueeze(0).unsqueeze(0)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32, device=self.device)
        filters['sobel_y'] = sobel_y.unsqueeze(0).unsqueeze(0)
        
        # Filtro de paso alto
        high_pass = torch.ones((3, 3), dtype=torch.float32, device=self.device) * -1
        high_pass[1, 1] = 8
        filters['high_pass'] = high_pass.unsqueeze(0).unsqueeze(0)
        
        # Filtro de paso bajo
        low_pass = torch.ones((3, 3), dtype=torch.float32, device=self.device) / 9
        filters['low_pass'] = low_pass.unsqueeze(0).unsqueeze(0)
        
        return filters
    
    def _ensure_tensor(self, data: Union[np.ndarray, torch.Tensor, List]) -> torch.Tensor:
        """
        Asegura que los datos estén en formato tensor de PyTorch.
        
        Args:
            data: Datos a convertir.
            
        Returns:
            Tensor de PyTorch.
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device)
        elif isinstance(data, list):
            return torch.tensor(data, device=self.device)
        else:
            raise ValueError(f"Tipo de datos no soportado: {type(data)}")
    
    def _ensure_2d(self, data: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        """
        Asegura que los datos estén en formato 2D adecuado para procesamiento óptico.
        
        Args:
            data: Tensor de entrada.
            dim: Dimensión objetivo (cuadrada).
            
        Returns:
            Tensor 2D.
        """
        # Convertir a float32
        data = data.to(dtype=torch.float32)
        
        # Aplanar si es necesario
        if data.dim() > 2:
            data = data.reshape(data.shape[0], -1)
        elif data.dim() == 1:
            data = data.unsqueeze(0)
        
        # Redimensionar a dimensión cuadrada si se especifica
        if dim is not None:
            if data.shape[0] * data.shape[1] > dim * dim:
                # Truncar si es demasiado grande
                data = data[:dim, :dim]
            else:
                # Rellenar con ceros si es demasiado pequeño
                padded = torch.zeros((dim, dim), dtype=torch.float32, device=self.device)
                padded[:data.shape[0], :data.shape[1]] = data
                data = padded
        
        return data
    
    def fourier_transform(self, data: Union[np.ndarray, torch.Tensor], inverse: bool = False) -> torch.Tensor:
        """
        Aplica transformada de Fourier a los datos.
        
        Args:
            data: Datos de entrada.
            inverse: Si es True, aplica transformada inversa.
            
        Returns:
            Transformada de Fourier (o su inversa) de los datos.
        """
        self.operations_count += 1
        self.last_operation_time = time.time()
        
        try:
            # Convertir a tensor
            tensor = self._ensure_tensor(data)
            
            # Asegurar formato 2D
            tensor = self._ensure_2d(tensor)
            
            # Aplicar transformada de Fourier
            if inverse:
                result = torch.fft.ifft2(tensor)
            else:
                result = torch.fft.fft2(tensor)
            
            return result
        except Exception as e:
            logger.error(f"Error en transformada de Fourier: {e}")
            # Devolver datos originales en caso de error
            return self._ensure_tensor(data)
    
    def convolve(self, data: Union[np.ndarray, torch.Tensor], kernel: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Aplica convolución a los datos con un kernel específico.
        
        Args:
            data: Datos de entrada.
            kernel: Kernel de convolución o nombre de filtro predefinido.
            
        Returns:
            Resultado de la convolución.
        """
        self.operations_count += 1
        self.last_operation_time = time.time()
        
        try:
            # Convertir a tensor
            tensor = self._ensure_tensor(data)
            
            # Asegurar formato 2D
            tensor = self._ensure_2d(tensor)
            
            # Preparar para convolución (añadir dimensiones de batch y canal)
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            
            # Obtener kernel
            if isinstance(kernel, str):
                if kernel in self.filters:
                    kernel_tensor = self.filters[kernel]
                else:
                    raise ValueError(f"Filtro no reconocido: {kernel}")
            else:
                kernel_tensor = self._ensure_tensor(kernel)
                if kernel_tensor.dim() == 2:
                    kernel_tensor = kernel_tensor.unsqueeze(0).unsqueeze(0)
            
            # Aplicar convolución
            result = F.conv2d(tensor, kernel_tensor, padding='same')
            
            # Eliminar dimensiones extra
            result = result.squeeze(0).squeeze(0)
            
            return result
        except Exception as e:
            logger.error(f"Error en convolución: {e}")
            # Devolver datos originales en caso de error
            return self._ensure_tensor(data)
    
    def correlate(self, data1: Union[np.ndarray, torch.Tensor], data2: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Calcula la correlación cruzada entre dos conjuntos de datos.
        
        Args:
            data1: Primer conjunto de datos.
            data2: Segundo conjunto de datos.
            
        Returns:
            Correlación cruzada.
        """
        self.operations_count += 1
        self.last_operation_time = time.time()
        
        try:
            # Convertir a tensores
            tensor1 = self._ensure_tensor(data1)
            tensor2 = self._ensure_tensor(data2)
            
            # Asegurar formato 2D
            tensor1 = self._ensure_2d(tensor1)
            tensor2 = self._ensure_2d(tensor2, dim=tensor1.shape[0])
            
            # Calcular correlación en dominio de frecuencia (más eficiente)
            fft1 = torch.fft.fft2(tensor1)
            fft2 = torch.fft.fft2(tensor2)
            
            # Multiplicación compleja conjugada
            correlation = torch.fft.ifft2(fft1 * torch.conj(fft2))
            
            # Tomar magnitud
            result = torch.abs(correlation)
            
            return result
        except Exception as e:
            logger.error(f"Error en correlación: {e}")
            # Devolver tensor de ceros en caso de error
            return torch.zeros_like(self._ensure_tensor(data1))
    
    def filter_frequency(self, data: Union[np.ndarray, torch.Tensor], filter_type: str, cutoff: float = 0.5) -> torch.Tensor:
        """
        Aplica filtrado en el dominio de frecuencia.
        
        Args:
            data: Datos de entrada.
            filter_type: Tipo de filtro ('low_pass', 'high_pass', 'band_pass', 'band_stop').
            cutoff: Frecuencia de corte normalizada (0-1) o tupla para filtros de banda.
            
        Returns:
            Datos filtrados.
        """
        self.operations_count += 1
        self.last_operation_time = time.time()
        
        try:
            # Convertir a tensor
            tensor = self._ensure_tensor(data)
            
            # Asegurar formato 2D
            tensor = self._ensure_2d(tensor)
            
            # Aplicar transformada de Fourier
            fft = torch.fft.fft2(tensor)
            fft_shift = torch.fft.fftshift(fft)
            
            # Crear máscara de filtro
            h, w = tensor.shape
            y, x = torch.meshgrid(
                torch.arange(-h//2, h//2, device=self.device),
                torch.arange(-w//2, w//2, device=self.device),
                indexing='ij'
            )
            
            # Distancia normalizada desde el centro
            d = torch.sqrt(x**2 + y**2) / (min(h, w) / 2)
            
            # Crear máscara según tipo de filtro
            if filter_type == 'low_pass':
                mask = d <= cutoff
            elif filter_type == 'high_pass':
                mask = d >= cutoff
            elif filter_type == 'band_pass':
                if not isinstance(cutoff, tuple) or len(cutoff) != 2:
                    cutoff = (0.3, 0.7)  # Valores por defecto
                mask = (d >= cutoff[0]) & (d <= cutoff[1])
            elif filter_type == 'band_stop':
                if not isinstance(cutoff, tuple) or len(cutoff) != 2:
                    cutoff = (0.3, 0.7)  # Valores por defecto
                mask = (d <= cutoff[0]) | (d >= cutoff[1])
            else:
                raise ValueError(f"Tipo de filtro no reconocido: {filter_type}")
            
            # Aplicar máscara
            filtered_fft = fft_shift * mask.to(torch.complex64)
            
            # Transformada inversa
            filtered_fft_ishift = torch.fft.ifftshift(filtered_fft)
            filtered = torch.fft.ifft2(filtered_fft_ishift)
            
            # Tomar parte real
            result = torch.real(filtered)
            
            return result
        except Exception as e:
            logger.error(f"Error en filtrado de frecuencia: {e}")
            # Devolver datos originales en caso de error
            return self._ensure_tensor(data)
    
    def holographic_transform(self, data: Union[np.ndarray, torch.Tensor], reference: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simula una transformación holográfica.
        
        Args:
            data: Datos de entrada (objeto).
            reference: Onda de referencia.
            
        Returns:
            Tupla (holograma, reconstrucción).
        """
        self.operations_count += 1
        self.last_operation_time = time.time()
        
        try:
            # Convertir a tensores
            object_wave = self._ensure_tensor(data)
            reference_wave = self._ensure_tensor(reference)
            
            # Asegurar formato 2D
            object_wave = self._ensure_2d(object_wave)
            reference_wave = self._ensure_2d(reference_wave, dim=object_wave.shape[0])
            
            # Normalizar amplitudes
            object_wave = object_wave / (torch.max(torch.abs(object_wave)) + 1e-8)
            reference_wave = reference_wave / (torch.max(torch.abs(reference_wave)) + 1e-8)
            
            # Convertir a números complejos si son reales
            if not object_wave.is_complex():
                object_wave = object_wave.to(torch.complex64)
            if not reference_wave.is_complex():
                reference_wave = reference_wave.to(torch.complex64)
            
            # Simular interferencia para crear holograma
            hologram = torch.abs(object_wave + reference_wave) ** 2
            
            # Simular reconstrucción (iluminación del holograma con onda de referencia)
            reconstruction = hologram * reference_wave
            
            # Aplicar transformada de Fourier para simular propagación
            reconstruction = torch.fft.ifft2(torch.fft.fft2(reconstruction))
            
            return hologram, reconstruction
        except Exception as e:
            logger.error(f"Error en transformación holográfica: {e}")
            # Devolver tensores de ceros en caso de error
            tensor = self._ensure_tensor(data)
            return torch.zeros_like(tensor), torch.zeros_like(tensor)
    
    def phase_contrast(self, data: Union[np.ndarray, torch.Tensor], phase_shift: float = np.pi/2) -> torch.Tensor:
        """
        Simula microscopía de contraste de fase.
        
        Args:
            data: Datos de entrada.
            phase_shift: Desplazamiento de fase a aplicar.
            
        Returns:
            Imagen con contraste de fase mejorado.
        """
        self.operations_count += 1
        self.last_operation_time = time.time()
        
        try:
            # Convertir a tensor
            tensor = self._ensure_tensor(data)
            
            # Asegurar formato 2D
            tensor = self._ensure_2d(tensor)
            
            # Convertir a complejo si es real
            if not tensor.is_complex():
                tensor = tensor.to(torch.complex64)
            
            # Aplicar transformada de Fourier
            fft = torch.fft.fft2(tensor)
            fft_shift = torch.fft.fftshift(fft)
            
            # Crear máscara de fase
            h, w = tensor.shape
            y, x = torch.meshgrid(
                torch.arange(-h//2, h//2, device=self.device),
                torch.arange(-w//2, w//2, device=self.device),
                indexing='ij'
            )
            
            # Aplicar desplazamiento de fase a componentes 
(Content truncated due to size limit. Use line ranges to read in chunks)