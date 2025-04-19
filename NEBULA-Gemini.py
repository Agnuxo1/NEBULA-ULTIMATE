
"""
NebulaAGI.py: The Unified Quantum-Optical AGI Simulation Framework

Version: 1.5 (Consolidated & Refined)
Author: Francisco Angulo de Lafuente (Concept) & AI Assistant (Implementation)
Date: 10 September 2024
Contact: (User's Contact Info if desired)

################################################################################
#  DISCLAIMER & IMPORTANT NOTES:                                             #
#------------------------------------------------------------------------------#
#  1. THIS IS NOT AGI/ASI: This program is a research prototype and simulation#
#     framework based on the user's novel concepts. It DOES NOT represent a  #
#     true Artificial General Intelligence or Superintelligence. Achieving   #
#     AGI/ASI is an unsolved scientific challenge.                           #
#  2. EXTREME COMPLEXITY: This system integrates many advanced concepts       #
#     (quantum simulation, optical physics metaphors, LLMs, evolution, self- #
#     modification). Its behavior is emergent and potentially unpredictable. #
#  3. SIMULATION LIMITATIONS: Quantum and optical elements are simulated on   #
#     classical hardware. This incurs significant computational cost and does#
#     not fully replicate real-world physics. Their benefit over optimized   #
#     classical approaches within this simulation is speculative.            #
#  4. HIGH RESOURCE USAGE: Running this code requires substantial CPU, RAM,  #
#     and potentially GPU resources (especially VRAM for LLMs).              #
#  5. STABILITY & DEBUGGING: Due to its complexity and self-modification    #
#     potential, stability is not guaranteed. Significant debugging and      #
#     tuning will likely be required. The self-modification feature is       #
#     experimental and potentially dangerous to the system's integrity.      #
#  6. USE AS A FRAMEWORK: Treat this code as a starting point for research   #
#     and experimentation with these integrated concepts.                    #
################################################################################

Description:
This program represents the unified core simulation framework for the NEBULA AGI project.
It integrates concepts from quantum mechanics (simulated circuits via PennyLane),
advanced optics (simulated light propagation for neuron interaction, conceptual
holographic memory via embeddings, placeholder optical computations via FFT),
dynamic neural networks in a spatial environment (NebulaSpace with QuantumNeurons),
evolutionary computation (DEAP), Large Language Model (LLM) integration
(Hugging Face Transformers), knowledge representation (Knowledge Graphs), and
rudimentary self-correction/improvement capabilities.

NebulaAGI aims to simulate an adaptable intelligence, learning from external
data and its own operational experience within this complex, simulated ecosystem.
"""

# ========================================================
# IMPORTS & INITIAL SETUP
# ========================================================
print("🐍 Loading Imports...")
import os, sys, time, logging, inspect, traceback, random, copy, tempfile, subprocess, importlib, re, json, math, gc, difflib, pickle, hashlib
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional, Union, Callable
from collections import deque, defaultdict

# --- Core ML/Simulation Libraries ---
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
# from torch.utils.tensorboard import SummaryWriter # Optional for detailed logging

import pennylane as qml
from pennylane import numpy as pnp # Use pnp consistently
from pennylane.optimize import AdamOptimizer
from pennylane.templates import StronglyEntanglingLayers, BasicEntanglerLayers

# --- Optional GPU Acceleration for Arrays ---
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("   INFO: CuPy found. GPU acceleration for array operations enabled.")
except ImportError:
    # print("   WARNING: CuPy not available, falling back to NumPy for array operations.")
    cp = np # Fallback to numpy
    CUPY_AVAILABLE = False

# --- Evolutionary Computation ---
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    print("   ERROR: DEAP library not found. Evolutionary features disabled. Install with 'pip install deap'")
    DEAP_AVAILABLE = False

# --- Hugging Face Transformers & Sentence Transformers ---
try:
    from huggingface_hub import login, HfApi, snapshot_download, whoami, hf_hub_download
    from transformers import (
        AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
        AutoModelForQuestionAnswering, AutoModel, pipeline, TrainingArguments, Trainer,
        ViTImageProcessor, BlipProcessor, BlipForConditionalGeneration, ViTForImageClassification,
        CLIPProcessor, CLIPModel, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer,
        set_seed, logging as hf_logging
    )
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
    hf_logging.set_verbosity_error() # Reduce Hugging Face logging noise
except ImportError:
    print("   ERROR: Transformers or sentence-transformers library not found. LLM and Embedding features disabled. "
          "Install with 'pip install transformers sentence-transformers huggingface_hub torch torchvision torchaudio'")
    TRANSFORMERS_AVAILABLE = False

# --- Knowledge Graph & Spatial Search ---
import networkx as nx
from scipy.spatial import cKDTree
import scipy.linalg

# --- NLP Tools ---
try:
    import spacy
    # nltk is used less directly now, but keep import if needed for specific functions
    # import nltk
    # from nltk.corpus import wordnet as wn
    NLP_AVAILABLE = True
except ImportError:
    print("   WARNING: SpaCy library not found. Advanced text processing features may be limited. Install with 'pip install spacy && python -m spacy download en_core_web_sm'")
    NLP_AVAILABLE = False

# --- Web/API Tools ---
import requests
from bs4 import BeautifulSoup
import wikipedia # For information gathering

# --- Utility & Monitoring ---
import psutil
try:
    import GPUtil # Optional for GPU monitoring
    GPUtil_AVAILABLE = True
except ImportError:
    # print("   INFO: GPUtil not found. Detailed GPU monitoring disabled.")
    GPUtil_AVAILABLE = False
from PIL import Image
import io

# --- Optional UI (PyQt6) ---
try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                                 QTextEdit, QLineEdit, QPushButton, QLabel, QMessageBox, QSplitter)
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
    from PyQt6.QtGui import QFont, QTextCursor
    PYQT_AVAILABLE = True
except ImportError:
    # print("   INFO: PyQt6 not found. GUI features disabled.")
    PYQT_AVAILABLE = False
    # Dummy classes if PyQt6 is not available
    class QMainWindow: pass
    class QWidget: pass
    class QApplication: pass
    class QThread: pass
    class pyqtSignal: def __init__(self, *args, **kwargs): pass
    class pyqtSlot: def __init__(self, *args): pass

# --- Optional Plotting ---
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    # print("   INFO: Matplotlib not found. Plotting features disabled.")
    MATPLOTLIB_AVAILABLE = False

# --- Bio-related imports (Optional, if functionality is added back) ---
# try:
#     from rdkit import Chem, RDLogger
#     from rdkit.Chem import AllChem
#     RDLogger.DisableLog("rdApp.*")
#     RDKIT_AVAILABLE = True
# except ImportError:
#     RDKIT_AVAILABLE = False
# try:
#     from Bio.PDB import PDBParser, Superimposer
#     BIOPYTHON_AVAILABLE = True
# except ImportError:
#     BIOPYTHON_AVAILABLE = False


# ========================================================
# GLOBAL CONFIGURATIONS & PARAMETERS
# ========================================================
print("⚙️ Configuring Parameters...")

PARAMETERS = {
    # --- General ---
    "SEED": 42,
    "DEVICE": None, # Will be set after checking CUDA
    "LOG_LEVEL": logging.INFO, # INFO, DEBUG, WARNING, ERROR
    "LOG_FILE": Path("./nebula_agi.log"),
    "ALLOW_INTERNET_ACCESS": True, # Allow access to Wikipedia, Hugging Face Hub etc.

    # --- NebulaSpace & QuantumNeurons ---
    "SPACE_DIMENSIONS": (100, 100, 100), # Spatial boundaries
    "INITIAL_NEURONS": 200, # Start with fewer neurons for faster testing
    "MAX_NEURONS": 1000,   # Limit neuron count
    "INPUT_DIM": 16, # Dimension of classical input affecting quantum state (keep small)
    "NUM_QUBITS": 4,    # Qubits per neuron (higher = exponentially slower simulation)
    "QUANTUM_LAYERS": 2, # Layers in the PennyLane circuit template
    "CONNECTION_PROBABILITY": 0.08, # Initial connection chance
    "MAX_CONNECTIONS": 10,        # Max connections per neuron
    "NEURON_INTERACTION_MODEL": "light_attenuation", # 'light_attenuation' or 'none'
    "LIGHT_ATTENUATION_FACTOR": 0.3, # How fast light fades with distance^2
    "LIGHT_RECEIVE_FACTOR": 0.02, # How much received light affects luminosity
    "LUMINOSITY_DECAY": 0.005,    # How much luminosity fades each step
    "NEURON_ACTIVITY_THRESHOLD": 0.1, # Luminosity threshold for pruning
    "NEURON_INACTIVITY_PERIOD": 100, # Iterations before inactive prune check
    "POSITION_UPDATE_DT": 0.1,   # Timestep for position updates
    "STRUCTURE_UPDATE_INTERVAL": 25, # How often to run clustering (iterations)
    "N_CLUSTERS": 5,              # Target clusters for K-Means

    # --- LLM & Knowledge ---
    "EMBEDDING_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2", # Efficient default
    "GENERATION_MODEL_SMALL": "gpt2", # Small, fast default for simple tasks
    "GENERATION_MODEL_LARGE": "microsoft/Phi-3-mini-4k-instruct", # More capable model (requires trust_remote_code=True)
    "CODEGEN_MODEL_NAME": "Salesforce/codegen-350M-mono", # Code generation model
    "QA_MODEL_NAME": "deepset/roberta-base-squad2", # Question Answering model
    "IMAGE_CAPTION_MODEL_NAME": "Salesforce/blip-image-captioning-base", # Image captioning
    "TRUST_REMOTE_CODE_LARGE_GEN": True, # Needed for models like Phi-3
    "MODEL_CACHE_DIR": Path("./model_cache"),
    "MAX_LLM_INPUT_LENGTH": 512, # Limit context for LLMs
    "MAX_LLM_GENERATION_LENGTH": 150, # Limit output length
    "MODEL_UNLOAD_DELAY": 900, # Unload inactive models after 15 mins (seconds)
    "KNOWLEDGE_GRAPH_FILE": Path("./nebula_kg.graphml"),
    "WIKIPEDIA_LANG": "en",
    "WIKIPEDIA_MAX_PAGES_PER_CYCLE": 2,
    "HOLOGRAPHIC_MEMORY_THRESHOLD": 0.65, # Similarity threshold for retrieval

    # --- Evolution (DEAP) ---
    "EVOLUTION_ENABLED": True if DEAP_AVAILABLE else False,
    "EVOLUTION_INTERVAL": 50, # Run evolution every N iterations
    "GA_POPULATION_SIZE": 20,
    "GA_GENERATIONS": 10, # Fewer generations per cycle for speed
    "GA_CXPB": 0.7, # Crossover probability
    "GA_MUTPB": 0.2, # Mutation probability
    "FITNESS_EVAL_TASKS": 3, # Number of tasks to run for fitness evaluation

    # --- Self-Modification ---
    "SELF_CORRECTION_ENABLED": True if TRANSFORMERS_AVAILABLE else False, # Requires CodeGen LLM
    "SELF_IMPROVEMENT_INTERVAL": 100, # Check for improvements every N iterations
    "CODE_EVALUATION_THRESHOLD": 0.5, # Confidence score to apply generated code change
    "MAX_ERROR_HISTORY": 50,
    "MAX_MODIFICATION_HISTORY": 20,

    # --- State & Backup ---
    "STATE_FILE": Path("./nebula_state.pkl"),
    "BACKUP_INTERVAL": 3600, # Backup state every hour (seconds)
    "BACKUP_DIRECTORY": Path("./nebula_backups"),

    # --- UI ---
    "UI_ENABLED": PYQT_AVAILABLE, # Enable UI only if PyQt6 is installed
}

# --- Dynamic Configuration ---
# Device Selection
if torch.cuda.is_available():
    PARAMETERS["DEVICE"] = torch.device("cuda")
    print("   INFO: CUDA Available. Using GPU.")
    torch.backends.cudnn.benchmark = True # Optimize convolution speed
else:
    PARAMETERS["DEVICE"] = torch.device("cpu")
    print("   INFO: CUDA not available. Using CPU.")
DEVICE = PARAMETERS["DEVICE"]

# Seed for Reproducibility
seed = PARAMETERS["SEED"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(seed)
pnp.random.seed(seed) # For Pennylane's numpy
if CUPY_AVAILABLE:
    cp.random.seed(seed)
if TRANSFORMERS_AVAILABLE:
    set_seed(seed)
print(f"   INFO: Random seeds set to {seed}")

# Create Directories
PARAMETERS["MODEL_CACHE_DIR"].mkdir(exist_ok=True)
PARAMETERS["BACKUP_DIRECTORY"].mkdir(exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(PARAMETERS["MODEL_CACHE_DIR"].absolute())
os.environ["HF_HOME"] = str(PARAMETERS["MODEL_CACHE_DIR"].absolute()) # Consolidate cache

# Configure Logging
log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] [%(name)-10s] %(message)s")
root_logger = logging.getLogger()
root_logger.setLevel(PARAMETERS["LOG_LEVEL"])
# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)
# File Handler
try:
    file_handler = logging.FileHandler(PARAMETERS["LOG_FILE"], mode='a') # Append mode
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
except Exception as e:
    print(f"   WARNING: Could not set up log file handler: {e}")
logger = logging.getLogger("NebulaAGI") # Logger specific to the main class

# --- Ray Initialization (Optional) ---
RAY_ENABLED = False # Default to disabled for simplicity/stability
# Uncomment to attempt Ray initialization
# try:
#     import ray
#     if not ray.is_initialized():
#         # Conservative resource allocation for Ray
#         num_cpus = max(1, psutil.cpu_count() // 2)
#         total_mem_gb = psutil.virtual_memory().total / (1024**3)
#         ray_mem_gb = max(2, int(total_mem_gb * 0.3)) # Use 30% of RAM, min 2GB
#         ray.init(num_cpus=num_cpus, object_store_memory=int(ray_mem_gb * 1024**3), ignore_reinit_error=True)
#         print(f"   INFO: Ray initialized successfully (CPUs: {num_cpus}, Memory: {ray_mem_gb:.2f} GB).")
#         RAY_ENABLED = True
#     else:
#         print("   INFO: Ray already initialized.")
#         RAY_ENABLED = True
# except ImportError:
#     print("   INFO: Ray library not found. Distributed computation disabled.")
# except Exception as e:
#     print(f"   WARNING: Failed to initialize Ray: {e}. Distributed computation disabled.")


# ========================================================
# DECORATORS
# ========================================================
print("🛠️ Defining Decorators...")

def safe_loop(max_retries=3, delay=5):
    """Decorator for safe loop handling with retries and exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                instance = args[0] if args and isinstance(args[0], NebulaAGI) else None
                try:
                    return func(*args, **kwargs)
                except KeyboardInterrupt:
                    logger.warning(f"🛑 User interrupt detected in {func.__name__}. Attempting graceful shutdown...")
                    if instance: instance.shutdown()
                    sys.exit(0)
                except Exception as e:
                    logger.error(f"🚨 Unhandled exception in {func.__name__} (Attempt {retries + 1}/{max_retries}): {e}", exc_info=True)
                    if instance and hasattr(instance, 'handle_error'):
                         try:
                             snippet = instance.get_relevant_code_snippet(depth=2)
                         except: snippet = "Could not retrieve code snippet."
                         instance.handle_error(f"Unhandled exception in {func.__name__}: {e}", snippet)

                    retries += 1
                    if retries < max_retries:
                        wait_time = delay * (2 ** (retries - 1))
                        logger.info(f"   Retrying {func.__name__} in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.critical(f"🔴 Maximum retries reached for {func.__name__}. Critical failure. Shutting down.")
                        if instance: instance.shutdown()
                        sys.exit(1) # Exit on persistent critical failure
            # This part should theoretically not be reached if max_retries > 0
            logger.critical(f"Exited retry loop for {func.__name__} unexpectedly.")
            if instance: instance.shutdown()
            sys.exit(1)
        return wrapper
    return decorator

def require_component(component_name: str):
    """Decorator to ensure a component (attribute) is initialized."""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            component = getattr(self, component_name, None)
            if component is None:
                logger.error(f"Operation '{func.__name__}' requires component '{component_name}', which is not initialized. Skipping.")
                # Optionally try to initialize it here if possible/safe
                return None # Indicate failure or skip
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def require_llm(model_key: str):
    """Decorator to ensure a specific LLM is loaded before use."""
    def decorator(func):
        @wraps(func) # Preserve original function metadata
        def wrapper(self: 'NebulaAGI', *args, **kwargs):
            if not TRANSFORMERS_AVAILABLE:
                logger.error(f"Transformers library not available. Cannot use LLM '{model_key}' for '{func.__name__}'.")
                return None
            if not self.is_llm_loaded(model_key):
                logger.info(f"LLM '{model_key}' required for '{func.__name__}' not loaded. Attempting to load...")
                if not self.load_llm_model(model_key):
                    logger.error(f"Failed to load required LLM '{model_key}'. Operation '{func.__name__}' cancelled.")
                    return None # Indicate failure
            # Ensure model is on the correct device
            model_info = self.llm_models.get(model_key)
            if model_info and model_info.get('model') and hasattr(model_info['model'], 'to'):
                 try:
                     model_info['model'].to(DEVICE)
                 except Exception as e:
                      logger.error(f"Failed to move model {model_key} to device {DEVICE}: {e}")
                      # Optionally unload the model if it can't be moved?
                      # self.unload_llm_model(model_key, force=True)
                      return None
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

# --- Function Wrapper (for preserving metadata) ---
from functools import wraps


# ========================================================
# UTILITY FUNCTIONS & CLASSES
# ========================================================
print("🛠️ Defining Utilities...")

def convert_to_numpy(data: Any) -> np.ndarray:
    """Converts PyTorch/CuPy tensors or lists/tuples to NumPy arrays."""
    if isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()
    elif CUPY_AVAILABLE and isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (list, tuple)):
        # Attempt to convert elements if they are tensors/arrays
        if data and isinstance(data[0], torch.Tensor):
             return np.array([convert_to_numpy(x) for x in data])
        else:
             return np.array(data) # Assume basic types
    elif isinstance(data, (int, float)):
        return np.array([data])
    else:
        logger.warning(f"Cannot reliably convert type {type(data)} to NumPy. Returning None.")
        return None

def calculate_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two numpy vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    # Ensure vectors are float type for dot product precision
    sim = np.dot(vec1.astype(float), vec2.astype(float)) / (norm1 * norm2)
    return float(np.clip(sim, -1.0, 1.0)) # Clip for numerical stability


# ========================================================
# CORE NEBULA COMPONENTS
# ========================================================

# --- Quantum Neuron ---
print("🧠 Defining QuantumNeuron...")
class QuantumNeuron(nn.Module):
    """Represents a quantum-inspired neuron in the NEBULA network."""
    def __init__(
        self,
        neuron_id: int,
        position: Union[List[float], np.ndarray, torch.Tensor],
        input_dim: int = PARAMETERS["INPUT_DIM"],
        num_qubits: int = PARAMETERS["NUM_QUBITS"],
        num_layers: int = PARAMETERS["QUANTUM_LAYERS"],
        device: torch.device = DEVICE
    ):
        super().__init__() # Initialize nn.Module
        self.id = neuron_id
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.input_dim = input_dim
        self._device = device # Use underscore to avoid conflict with nn.Module's device property

        # --- Quantum Circuit Setup ---
        try:
            self.qdevice = qml.device("default.qubit", wires=self.num_qubits)
        except Exception as e:
            logger.error(f"Neuron {self.id}: Failed to initialize Pennylane device: {e}", exc_info=True)
            self.qdevice = None
            self.quantum_circuit = None

        # Quantum circuit parameters (Torch Parameters for autograd)
        weight_shapes = StronglyEntanglingLayers.shape(n_layers=self.num_layers, n_wires=self.num_qubits)
        self.qlayer_weights = nn.Parameter(torch.rand(weight_shapes, device=self._device) * 2 * torch.pi * 0.1)

        if self.qdevice:
            # Define the quantum node using the class instance's device and weights
            @qml.qnode(self.qdevice, interface="torch", diff_method="parameter-shift")
            def quantum_circuit_node(inputs, weights):
                # Encoding: Map classical input to rotation angles
                # Ensure inputs is a 1D tensor
                inputs = inputs.flatten()
                input_len = inputs.shape[0]
                # Repeat or truncate inputs to match num_qubits for simple angle encoding
                angles = inputs.repeat( (self.num_qubits + input_len - 1) // input_len )[:self.num_qubits] * torch.pi

                for i in range(self.num_qubits):
                    qml.RY(angles[i], wires=i) # Encode input

                # Entangling Layers
                StronglyEntanglingLayers(weights, wires=range(self.num_qubits))

                # Measurement: Expectation value of PauliZ on each qubit
                return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

            self.quantum_circuit = quantum_circuit_node
        else:
            logger.warning(f"Neuron {self.id}: Quantum circuit disabled due to device init failure.")

        # --- Classical Properties ---
        # Ensure position is a Parameter on the correct device
        if not isinstance(position, torch.Tensor):
             pos_np = convert_to_numpy(position) # Handles list, np.ndarray
             if pos_np is None: raise ValueError("Invalid position type for Neuron")
             position_tensor = torch.tensor(pos_np, dtype=torch.float32, device=self._device)
        else:
            position_tensor = position.to(dtype=torch.float32, device=self._device)

        self.position = nn.Parameter(position_tensor) # Learnable position

        # Luminosity (also learnable potentially, or just state)
        self.luminosity = nn.Parameter(torch.rand(1, device=self._device) * 0.3 + 0.1) # Start dimmer

        # Connections (Store tuples: (target_neuron_id, strength))
        self.connections: List[Tuple[int, float]] = []
        self.last_activity_time = time.time()
        self.cluster_id: Optional[str] = None
        self.sector_id: Optional[str] = None

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass: Processes classical input through the quantum circuit (if available)
        and combines with classical state.
        Returns a tensor representing the neuron's output/state.
        """
        # --- Input Processing ---
        if isinstance(x, np.ndarray):
            input_tensor = torch.from_numpy(x).float().to(self._device)
        elif isinstance(x, torch.Tensor):
            input_tensor = x.float().to(self._device)
        else:
            logger.error(f"Neuron {self.id}: Invalid input type {type(x)}. Expected NumPy array or Torch tensor.")
            # Return a zero tensor matching the expected output dimension (num_qubits)
            return torch.zeros(self.num_qubits, device=self._device)

        # Reshape/pad/truncate input to match self.input_dim
        input_tensor = input_tensor.flatten()
        current_dim = input_tensor.shape[0]
        if current_dim > self.input_dim:
            input_tensor = input_tensor[:self.input_dim]
        elif current_dim < self.input_dim:
            padding = torch.zeros(self.input_dim - current_dim, device=self._device)
            input_tensor = torch.cat((input_tensor, padding))

        # --- Quantum Processing ---
        if self.quantum_circuit:
            try:
                q_output = self.quantum_circuit(input_tensor, self.qlayer_weights)
                # Stack the list of expectation values into a tensor
                quantum_state = torch.stack(q_output)
            except Exception as e:
                logger.error(f"Neuron {self.id}: Error during quantum circuit execution: {e}", exc_info=False) # Avoid excessive logging
                quantum_state = torch.zeros(self.num_qubits, device=self._device) # Fallback
        else:
            # Fallback classical behavior if quantum circuit is disabled
            # Simple linear layer acting on input, output dim = num_qubits
            # Create a linear layer on the fly if needed (less efficient)
            # linear_fallback = nn.Linear(self.input_dim, self.num_qubits, device=self._device)
            # quantum_state = torch.tanh(linear_fallback(input_tensor))
            # Or just return zeros/random to indicate failure/no processing
            quantum_state = torch.zeros(self.num_qubits, device=self._device)


        # --- Combine Quantum & Classical State ---
        # Example: Scale quantum output by luminosity (or other combinations)
        final_output = quantum_state * self.luminosity.clamp(0.0, 1.0)

        self.last_activity_time = time.time()
        return final_output

    def emit_light(self) -> float:
        """Calculates the base light intensity emitted by the neuron."""
        # Simple model: proportional to luminosity
        return self.luminosity.item()

    def receive_light(self, intensity: float):
        """Processes received light, updating internal state (e.g., luminosity)."""
        update_factor = PARAMETERS.get("LIGHT_RECEIVE_FACTOR", 0.05)
        # Use data attribute for in-place update of parameter tensor
        self.luminosity.data += intensity * update_factor
        self.luminosity.data.clamp_(0.0, 1.0) # Clamp luminosity between 0 and 1
        self.last_activity_time = time.time()

    def decay_luminosity(self):
        """Applies decay factor to luminosity."""
        decay_rate = PARAMETERS.get("LUMINOSITY_DECAY", 0.01)
        self.luminosity.data *= (1.0 - decay_rate)
        self.luminosity.data.clamp_(0.0, 1.0)

    def get_state(self) -> Dict[str, Any]:
        """Returns the serializable state of the neuron."""
        state = {
            "id": self.id,
            "position": convert_to_numpy(self.position.data), # Store data part
            "luminosity": self.luminosity.item(),
            "qlayer_weights": convert_to_numpy(self.qlayer_weights.data),
            "connections": copy.deepcopy(self.connections), # Deep copy list of tuples
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
        """Creates a neuron instance from a saved state."""
        neuron = cls(
            neuron_id=state["id"],
            position=state["position"], # Position will be converted to tensor in __init__
            input_dim=state.get("input_dim", PARAMETERS["INPUT_DIM"]),
            num_qubits=state.get("num_qubits", PARAMETERS["NUM_QUBITS"]),
            num_layers=state.get("num_layers", PARAMETERS["QUANTUM_LAYERS"]),
            device=device
        )
        # Load parameters directly into the nn.Parameter tensors
        neuron.luminosity.data = torch.tensor([state["luminosity"]], device=device)
        if 'qlayer_weights' in state and state['qlayer_weights'] is not None:
             neuron.qlayer_weights.data = torch.tensor(state["qlayer_weights"], dtype=torch.float32, device=device)
        # Ensure position is loaded correctly
        neuron.position.data = torch.tensor(state["position"], dtype=torch.float32, device=device)

        neuron.connections = state.get("connections", [])
        neuron.last_activity_time = state.get("last_activity_time", time.time())
        neuron.cluster_id = state.get("cluster_id")
        neuron.sector_id = state.get("sector_id")
        return neuron

    def get_embedding(self) -> np.ndarray:
        """Gets a feature vector representing the neuron's state for clustering."""
        pos = convert_to_numpy(self.position.data)
        lum = self.luminosity.item()
        # Use a stable aggregate of quantum weights (e.g., norm or mean)
        weights_agg = np.linalg.norm(convert_to_numpy(self.qlayer_weights.data)) if self.qlayer_weights is not None else 0.0
        # Add number of connections?
        num_conn = len(self.connections)

        # Ensure consistent length and type
        embedding = np.array([pos[0], pos[1], pos[2], lum, weights_agg, num_conn], dtype=np.float32)
        return embedding

# --- Cluster & Sector (Simplified Structure Holders) ---
print("🏗️ Defining Cluster & Sector...")
class Cluster:
    """Represents a cluster of neurons."""
    def __init__(self, cluster_id: str, position: torch.Tensor, device: torch.device):
        self.id = cluster_id
        self.position = position.to(device) # Center of the cluster
        self.neuron_ids: set[int] = set()
        self.device = device

    def add_neuron(self, neuron: QuantumNeuron):
        self.neuron_ids.add(neuron.id)
        neuron.cluster_id = self.id

    def remove_neuron(self, neuron: QuantumNeuron):
        self.neuron_ids.discard(neuron.id)
        if neuron.cluster_id == self.id: # Avoid clearing if reassigned already
            neuron.cluster_id = None

    def update_position(self, neurons_dict: Dict[int, QuantumNeuron]):
        """Updates cluster position based on its neurons' centroids."""
        if not self.neuron_ids: return
        positions = [neurons_dict[nid].position.data for nid in self.neuron_ids if nid in neurons_dict]
        if positions:
            self.position = torch.mean(torch.stack(positions), dim=0)

    def get_state(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "position": convert_to_numpy(self.position),
            "neuron_ids": list(self.neuron_ids)
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any], device: torch.device) -> "Cluster":
        cluster = cls(
            cluster_id=state["id"],
            position=torch.tensor(state["position"], dtype=torch.float32, device=device),
            device=device
        )
        cluster.neuron_ids = set(state.get("neuron_ids", []))
        return cluster

class Sector:
    """Represents a sector containing multiple clusters."""
    def __init__(self, sector_id: str, position: torch.Tensor, device: torch.device):
        self.id = sector_id
        self.position = position.to(device)
        self.cluster_ids: set[str] = set()
        self.device = device

    def add_cluster(self, cluster: Cluster):
        self.cluster_ids.add(cluster.id)
        # Assign sector ID to neurons in the cluster (handled by NebulaSpace typically)

    def remove_cluster(self, cluster: Cluster):
        self.cluster_ids.discard(cluster.id)

    def update_position(self, clusters_dict: Dict[str, Cluster]):
        """Updates sector position based on its clusters' centroids."""
        if not self.cluster_ids: return
        positions = [clusters_dict[cid].position for cid in self.cluster_ids if cid in clusters_dict]
        if positions:
            self.position = torch.mean(torch.stack(positions), dim=0)

    def get_state(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "position": convert_to_numpy(self.position),
            "cluster_ids": list(self.cluster_ids)
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any], device: torch.device) -> "Sector":
        sector = cls(
            sector_id=state["id"],
            position=torch.tensor(state["position"], dtype=torch.float32, device=device),
            device=device
        )
        sector.cluster_ids = set(state.get("cluster_ids", []))
        return sector

# --- Nebula Space (Environment) ---
print("🌌 Defining NebulaSpace...")
class NebulaSpace:
    """Manages the dynamic spatial environment for NEBULA neurons, clusters, and sectors."""
    def __init__(
        self,
        dimensions: Tuple[int, int, int] = PARAMETERS["SPACE_DIMENSIONS"],
        device: torch.device = DEVICE,
    ):
        logger.info(f"Initializing NebulaSpace on {device} with dimensions {dimensions}")
        self.dimensions = torch.tensor(dimensions, dtype=torch.float32, device=device)
        self.device = device
        # Use dictionaries for efficient lookup by ID
        self.neurons: Dict[int, QuantumNeuron] = {}
        self.clusters: Dict[str, Cluster] = {}
        self.sectors: Dict[str, Sector] = {}
        self.next_neuron_id = 0
        self.next_cluster_id = 0
        self.next_sector_id = 0
        self.kdtree = None # For efficient nearest neighbor search
        self.kdtree_needs_update = True
        # Use NetworkX for explicit connections (can store strength as edge weight)
        self.connection_graph = nx.Graph()

    def _get_new_id(self, type: str) -> Union[int, str]:
        """Generates a unique ID for neurons, clusters, or sectors."""
        if type == "neuron":
            self.next_neuron_id += 1
            return self.next_neuron_id - 1
        elif type == "cluster":
            self.next_cluster_id += 1
            return f"C{self.next_cluster_id - 1}"
        elif type == "sector":
            self.next_sector_id += 1
            return f"S{self.next_sector_id - 1}"
        else:
            raise ValueError(f"Unknown ID type requested: {type}")

    def add_neuron(self, neuron: QuantumNeuron, assign_structure=True):
        """Adds a neuron to the space, optionally assigning it to structures."""
        if neuron.id in self.neurons:
            logger.warning(f"Neuron {neuron.id} already exists. Skipping add.")
            return
        if not isinstance(neuron, QuantumNeuron):
             logger.error(f"Attempted to add non-QuantumNeuron object with ID {neuron.id}")
             return

        self.neurons[neuron.id] = neuron
        self.connection_graph.add_node(neuron.id) # Add node to graph
        self.kdtree_needs_update = True

        if assign_structure:
            self._assign_neuron_to_structure(neuron)

    def remove_neuron(self, neuron_id: int):
        """Removes a neuron and cleans up graph and structures."""
        if neuron_id not in self.neurons:
            logger.warning(f"Neuron {neuron_id} not found for removal.")
            return

        neuron = self.neurons.pop(neuron_id)
        logger.debug(f"Removing neuron {neuron_id}")

        # Remove from graph
        if self.connection_graph.has_node(neuron_id):
            # Remove edges connected to this node first
            edges_to_remove = list(self.connection_graph.edges(neuron_id))
            self.connection_graph.remove_edges_from(edges_to_remove)
            # Remove the node itself
            self.connection_graph.remove_node(neuron_id)

        # Update internal connection lists of other neurons
        for other_neuron in self.neurons.values():
            other_neuron.connections = [(target_id, strength) for target_id, strength in other_neuron.connections if target_id != neuron_id]

        # Remove from Cluster
        if neuron.cluster_id and neuron.cluster_id in self.clusters:
            cluster = self.clusters[neuron.cluster_id]
            cluster.remove_neuron(neuron)
            if not cluster.neuron_ids: # If cluster becomes empty
                self._remove_empty_cluster(cluster.id)

        # Remove from Sector (should be handled by removing cluster)
        neuron.sector_id = None # Clear sector ref just in case

        self.kdtree_needs_update = True

    def _remove_empty_cluster(self, cluster_id: str):
        """Removes an empty cluster and potentially its parent sector if empty."""
        if cluster_id not in self.clusters: return
        cluster = self.clusters.pop(cluster_id)
        logger.info(f"Removing empty cluster {cluster_id}")

        # Find the sector containing this cluster and remove reference
        found_sector = None
        for sector in self.sectors.values():
            if cluster_id in sector.cluster_ids:
                sector.remove_cluster(cluster)
                found_sector = sector
                break

        if found_sector and not found_sector.cluster_ids:
             self._remove_empty_sector(found_sector.id)

    def _remove_empty_sector(self, sector_id: str):
        """Removes an empty sector."""
        if sector_id in self.sectors:
             del self.sectors[sector_id]
             logger.info(f"Removing empty sector {sector_id}")

    def _assign_neuron_to_structure(self, neuron: QuantumNeuron):
        """Assigns a neuron to the nearest cluster/sector, creating new ones if needed."""
        if not self.sectors: # Create first sector if none exist
            sector_id = self._get_new_id("sector")
            # Place sector near neuron
            sector_pos = neuron.position.data + (torch.rand_like(neuron.position.data) - 0.5) * 10
            sector = Sector(sector_id, sector_pos.clamp(0, self.dimensions - 1), self.device)
            self.sectors[sector_id] = sector
            logger.debug(f"Created first sector {sector_id} for neuron {neuron.id}")

        nearest_sector = self._find_nearest_structure(neuron.position.data, self.sectors)
        if not nearest_sector:
             # This shouldn't happen if the block above works, but as a fallback
             logger.warning("Could not find nearest sector, creating fallback.")
             sector_id = self._get_new_id("sector")
             sector = Sector(sector_id, neuron.position.data.clone(), self.device)
             self.sectors[sector_id] = sector
             nearest_sector = sector

        neuron.sector_id = nearest_sector.id

        # Find or create cluster within the sector
        clusters_in_sector = {cid: self.clusters[cid] for cid in nearest_sector.cluster_ids if cid in self.clusters}
        if not clusters_in_sector: # Create first cluster in this sector
            cluster_id = self._get_new_id("cluster")
            cluster = Cluster(cluster_id, neuron.position.data.clone(), self.device)
            self.clusters[cluster_id] = cluster
            nearest_sector.add_cluster(cluster)
            logger.debug(f"Created first cluster {cluster_id} in sector {nearest_sector.id}")
            nearest_cluster = cluster
        else:
            nearest_cluster = self._find_nearest_structure(neuron.position.data, clusters_in_sector)
            if not nearest_cluster: # Should always find one if clusters_in_sector is not empty
                 logger.error("Logic error: Could not find nearest cluster among non-empty set.")
                 # Fallback: assign to first cluster in sector
                 nearest_cluster = next(iter(clusters_in_sector.values()))


        nearest_cluster.add_neuron(neuron)
        # logger.debug(f"Assigned neuron {neuron.id} to cluster {nearest_cluster.id}, sector {nearest_sector.id}")

    def _find_nearest_structure(self, position: torch.Tensor, structure_dict: Dict[str, Union[Cluster, Sector]]) -> Optional[Union[Cluster, Sector]]:
        """Finds the structure (Cluster or Sector) closest to the position."""
        if not structure_dict: return None
        min_dist_sq = float('inf')
        nearest_struct = None
        for struct in structure_dict.values():
            dist_sq = torch.sum((position - struct.position)**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_struct = struct
        return nearest_struct

    def _build_kdtree(self):
        """Builds/updates the KDTree for efficient neighbor searches."""
        if not self.neurons:
            self.kdtree = None
            self.kdtree_neuron_ids = []
            return

        positions_np = [convert_to_numpy(n.position.data) for n in self.neurons.values()]
        # Filter out any None positions if conversion failed
        valid_positions = [p for p in positions_np if p is not None]
        valid_ids = [nid for nid, n in self.neurons.items() if convert_to_numpy(n.position.data) is not None]


        if not valid_positions:
             self.kdtree = None
             self.kdtree_neuron_ids = []
             return

        try:
            positions_array = np.array(valid_positions)
            # Check for NaNs or Infs just in case
            if np.any(np.isnan(positions_array)) or np.any(np.isinf(positions_array)):
                 logger.error("NaN or Inf detected in neuron positions! Cannot build KDTree.")
                 # Attempt to fix or remove offending neurons? For now, disable KDTree.
                 self.kdtree = None
                 self.kdtree_neuron_ids = []
                 # Identify offenders:
                 # offenders = [nid for nid, pos in zip(valid_ids, valid_positions) if np.any(np.isnan(pos)) or np.any(np.isinf(pos))]
                 # logger.error(f"Offending neuron IDs: {offenders}")
            else:
                 self.kdtree = cKDTree(positions_array)
                 self.kdtree_neuron_ids = valid_ids # Store IDs corresponding to KDTree points
                 self.kdtree_needs_update = False
                 # logger.debug("KDTree updated.")
        except Exception as e:
            logger.error(f"Error building KDTree: {e}", exc_info=True)
            self.kdtree = None # Disable KDTree on error


    def find_neurons_within_radius(self, position: torch.Tensor, radius: float) -> List[QuantumNeuron]:
        """Finds neurons within a given radius using KDTree if available."""
        if self.kdtree_needs_update or self.kdtree is None:
            self._build_kdtree()

        target_pos_np = convert_to_numpy(position)
        if target_pos_np is None: return [] # Cannot search if position is invalid

        found_neurons = []

        if self.kdtree and self.kdtree_neuron_ids:
            try:
                 # query_ball_point returns indices into the data array used to build the tree
                 indices = self.kdtree.query_ball_point(target_pos_np, r=radius, p=2) # p=2 for Euclidean distance
                 for idx in indices:
                     # Check if index is valid for our ID list
                     if 0 <= idx < len(self.kdtree_neuron_ids):
                         neuron_id = self.kdtree_neuron_ids[idx]
                         if neuron_id in self.neurons: # Check neuron still exists
                             found_neurons.append(self.neurons[neuron_id])
                     else:
                         logger.warning(f"KDTree returned invalid index {idx}. Tree might be stale.")

            except Exception as e:
                logger.error(f"Error querying KDTree: {e}. Falling back to linear scan.", exc_info=False)
                self.kdtree = None # Disable KDTree on error
                # Fallback to linear scan (will execute below)
        else:
            # Linear scan if KDTree is not available or failed
            for neuron_id, neuron in self.neurons.items():
                dist = torch.norm(neuron.position.data - position)
                if dist <= radius:
                    found_neurons.append(neuron)

        return found_neurons


    def calculate_light_intensity(self, emitter: QuantumNeuron, receiver_pos: torch.Tensor) -> float:
        """Calculates light intensity at receiver position from an emitter neuron."""
        distance_sq = torch.sum((emitter.position.data - receiver_pos)**2)
        # Add small epsilon to avoid division by zero and extreme intensity at zero distance
        attenuation = 1.0 / (distance_sq + 1e-5)
        # Use emitter's current luminosity
        intensity = emitter.emit_light() * attenuation * PARAMETERS.get("LIGHT_ATTENUATION_FACTOR", 0.3)
        # Clamp intensity to a reasonable maximum
        return torch.clamp(intensity, 0.0, 5.0).item() # Return as float

    # --- Simulation Steps ---

    def run_simulation_step(self, iteration: int):
        """Runs one step of the Nebula simulation."""
        if not self.neurons:
             logger.warning("No neurons in NebulaSpace. Skipping simulation step.")
             return

        start_time = time.time()
        logger.debug(f"--- Starting Simulation Step {iteration} ---")

        # 1. Propagate Light & Update Neuron States (Receive Light)
        self._propagate_light_step()

        # 2. Update Connections (Prune weak, Add new based on proximity/activity?)
        self._update_connections_step()

        # 3. Update Neuron Positions (based on interaction model)
        self._update_positions_step()

        # 4. Internal Neuron Forward Pass (Optional - if neurons process internal state)
        # self._run_neuron_forward_pass() # Decide if this is needed per step

        # 5. Decay Luminosity & Prune Inactive Neurons
        self._decay_and_prune_step(iteration)

        # 6. Update Structure (Clustering, Merging) - Less frequently
        if iteration % PARAMETERS.get("STRUCTURE_UPDATE_INTERVAL", 25) == 0:
            self.update_structure()

        # Mark KDTree for update if positions changed significantly
        # self.kdtree_needs_update = True # Assume positions always change slightly

        elapsed = time.time() - start_time
        logger.debug(f"--- Simulation Step {iteration} finished in {elapsed:.4f} seconds ---")

    def _propagate_light_step(self):
        """Calculates and applies light intensity received by each neuron."""
        num_neurons = len(self.neurons)
        if num_neurons < 2: return

        # Store intensities to apply updates simultaneously after calculation
        received_intensities = defaultdict(float)
        all_neurons_list = list(self.neurons.values()) # Fixed list for this step

        # Simple N^2 iteration (can be optimized with KDTree for sparse interactions)
        for i in range(num_neurons):
            emitter = all_neurons_list[i]
            # Find potential receivers within a certain radius? Might be faster.
            # For now, calculate for all pairs.
            for j in range(num_neurons):
                if i == j: continue # Neuron doesn't illuminate itself this way
                receiver = all_neurons_list[j]
                intensity = self.calculate_light_intensity(emitter, receiver.position.data)
                received_intensities[receiver.id] += intensity

        # Apply updates based on accumulated intensities
        for neuron_id, total_intensity in received_intensities.items():
            if neuron_id in self.neurons:
                self.neurons[neuron_id].receive_light(total_intensity)
        # logger.debug("Light propagation finished.")

    def _update_connections_step(self):
        """Updates connections between neurons (in internal list and graph)."""
        # logger.debug("Updating connections...")
        max_connections = PARAMETERS.get("MAX_CONNECTIONS", 10)
        connection_prob = PARAMETERS.get("CONNECTION_PROBABILITY", 0.08)
        connection_radius = 30 # Max distance for potential connections (adjust based on space size)
        weak_connection_threshold = 0.05 # Strength below which connections are pruned

        edges_to_add = []
        edges_to_remove = []
        edges_to_update = {} # Store { (u,v): new_weight }

        for neuron1_id, neuron1 in self.neurons.items():
            # --- Prune existing weak or invalid connections ---
            current_connections_map = {target_id: strength for target_id, strength in neuron1.connections}
            valid_targets_in_list = set(current_connections_map.keys())
            connections_to_keep = []
            for target_id, strength in neuron1.connections:
                 if target_id not in self.neurons: # Target removed
                     edges_to_remove.append(tuple(sorted((neuron1_id, target_id))))
                     valid_targets_in_list.remove(target_id)
                     continue
                 # Example decay:
                 new_strength = strength * 0.99
                 if new_strength < weak_connection_threshold:
                     edges_to_remove.append(tuple(sorted((neuron1_id, target_id))))
                     valid_targets_in_list.remove(target_id)
                 else:
                     connections_to_keep.append((target_id, new_strength))
                     # Check if edge exists before updating weight (graph might be slightly out of sync)
                     # if self.connection_graph.has_edge(neuron1_id, target_id):
                     edges_to_update[tuple(sorted((neuron1_id, target_id)))] = new_strength


            neuron1.connections = connections_to_keep
            num_current_connections = len(neuron1.connections)

            # --- Add new connections ---
            if num_current_connections < max_connections:
                 # Find potential new neighbors within radius
                 potential_neighbors = self.find_neurons_within_radius(neuron1.position.data, connection_radius)
                 # Filter out self and already connected neurons
                 eligible_targets = [
                     n for n in potential_neighbors
                     if n.id != neuron1_id and n.id not in valid_targets_in_list
                 ]

                 # Add new connections probabilistically
                 added_count = 0
                 random.shuffle(eligible_targets) # Avoid bias towards closer neurons if many candidates
                 for target_neuron in eligible_targets:
                      if num_current_connections + added_count >= max_connections: break
                      if random.random() < connection_prob:
                           # Calculate initial strength (e.g., based on inverse distance)
                           dist = torch.norm(neuron1.position.data - target_neuron.position.data).item()
                           strength = max(0.1, (connection_radius - dist) / connection_radius) # Example: linear decay
                           strength = round(strength, 3) # Keep precision reasonable

                           neuron1.connections.append((target_neuron.id, strength))
                           edges_to_add.append((neuron1_id, target_neuron.id, {"weight": strength}))
                           added_count += 1

        # --- Batch update the NetworkX graph ---
        # Remove duplicates before modifying graph
        unique_edges_to_remove = set(edges_to_remove)
        # logger.debug(f"Removing {len(unique_edges_to_remove)} edges from graph.")
        self.connection_graph.remove_edges_from(list(unique_edges_to_remove))

        # logger.debug(f"Adding {len(edges_to_add)} edges to graph.")
        self.connection_graph.add_edges_from(edges_to_add)

        # logger.debug(f"Updating weights for {len(edges_to_update)} edges.")
        for (u, v), weight in edges_to_update.items():
             # Ensure edge still exists after removals before updating
             if self.connection_graph.has_edge(u, v):
                 self.connection_graph[u][v]['weight'] = weight
        # logger.debug("Connection update finished.")


    def _update_positions_step(self):
        """Updates neuron positions based on the interaction model."""
        model = PARAMETERS.get("NEURON_INTERACTION_MODEL", "none")
        if model == "none" or not self.neurons:
            return

        # logger.debug(f"Updating positions using model: {model}")
        forces = {nid: torch.zeros(3, device=self.device) for nid in self.neurons}
        # Get fixed list for iteration (positions might change during calculation otherwise)
        current_positions = {nid: n.position.data.clone() for nid, n in self.neurons.items()}
        neuron_list = list(self.neurons.values())

        if model == "light_attenuation":
            attraction_factor = 0.01 # Pulls together if strongly interacting
            repulsion_factor = 0.03  # Pushes apart otherwise
            intensity_threshold = 0.2 # Threshold to switch between attraction/repulsion

            for i in range(len(neuron_list)):
                 neuron1 = neuron_list[i]
                 pos1 = current_positions[neuron1.id]
                 for j in range(i + 1, len(neuron_list)):
                     neuron2 = neuron_list[j]
                     pos2 = current_positions[neuron2.id]

                     direction = pos2 - pos1
                     distance_sq = torch.sum(direction**2)
                     distance = torch.sqrt(distance_sq + 1e-6) # Avoid division by zero

                     # Calculate mutual intensity (simplification)
                     intensity1 = self.calculate_light_intensity(neuron1, pos2)
                     intensity2 = self.calculate_light_intensity(neuron2, pos1)
                     mutual_intensity = (intensity1 + intensity2) / 2.0

                     if mutual_intensity > intensity_threshold:
                         # Attraction force proportional to intensity and distance
                         force_magnitude = attraction_factor * mutual_intensity * distance
                     else:
                         # Repulsion force inversely proportional to distance squared
                         force_magnitude = -repulsion_factor / distance_sq

                     force_vector = force_magnitude * (direction / distance)

                     # Clamp force magnitude to prevent extreme velocities
                     force_vector = torch.clamp(force_vector, -0.5, 0.5)

                     forces[neuron1.id] += force_vector
                     forces[neuron2.id] -= force_vector # Newton's third law

        # --- Apply forces and update positions ---
        dt = PARAMETERS.get("POSITION_UPDATE_DT", 0.1)
        max_displacement_per_step = 0.5 # Limit movement speed

        positions_changed = False
        for neuron_id, force in forces.items():
             if neuron_id in self.neurons:
                 neuron = self.neurons[neuron_id]
                 # Simple Euler integration: velocity = force * dt (mass=1)
                 velocity = force * dt
                 # Clamp velocity
                 velocity_norm = torch.norm(velocity)
                 if velocity_norm > max_displacement_per_step:
                     velocity = velocity * (max_displacement_per_step / velocity_norm)

                 if torch.any(torch.abs(velocity) > 1e-4): # Only update if movement is significant
                    # Update position parameter's data in-place
                    neuron.position.data += velocity
                    # Keep neurons within bounds (bounce or clamp) - Simple clamp:
                    neuron.position.data.clamp_(min=torch.zeros(3, device=self.device), max=self.dimensions - 1)
                    positions_changed = True

        if positions_changed:
             self.kdtree_needs_update = True
             # logger.debug("Position update finished.")


    def _decay_and_prune_step(self, iteration: int):
        """Decays luminosity and removes inactive neurons."""
        # logger.debug("Decaying luminosity and pruning...")
        current_time = time.time()
        inactive_threshold = PARAMETERS.get("NEURON_ACTIVITY_THRESHOLD", 0.1)
        inactivity_period = PARAMETERS.get("NEURON_INACTIVITY_PERIOD", 100) # In iterations
        neurons_to_prune = []

        # Check only periodically to save computation
        if iteration % 20 != 0: # Check every 20 iterations
             # Still decay luminosity every step
             for neuron in self.neurons.values():
                 neuron.decay_luminosity()
             return

        logger.debug("Performing periodic prune check...")
        for neuron_id, neuron in self.neurons.items():
            neuron.decay_luminosity()
            # Pruning logic (based on luminosity AND last activity iteration count?)
            # We use time here, maybe switch to iterations?
            # if neuron.luminosity.item() < inactive_threshold and \
            #    (current_time - neuron.last_activity_time) > inactivity_period_secs: # If using time
            # Check based on iterations is harder unless we store last active iteration

            # Simpler: Prune if luminosity is very low for a while
            if neuron.luminosity.item() < inactive_threshold:
                neurons_to_prune.append(neuron_id)


        if neurons_to_prune and len(self.neurons) > PARAMETERS.get("MIN_NEURONS", 50): # Don't prune below minimum
             num_to_prune = min(len(neurons_to_prune), len(self.neurons) - PARAMETERS.get("MIN_NEURONS", 50))
             # Prune the lowest luminosity ones among the candidates
             candidates = {nid: self.neurons[nid].luminosity.item() for nid in neurons_to_prune}
             sorted_candidates = sorted(candidates.items(), key=lambda item: item[1])
             ids_to_remove = [nid for nid, lum in sorted_candidates[:num_to_prune]]

             if ids_to_remove:
                 logger.info(f"Pruning {len(ids_to_remove)} inactive neurons (luminosity < {inactive_threshold:.2f}).")
                 for neuron_id in ids_to_remove:
                      self.remove_neuron(neuron_id) # Handles graph/structure cleanup

        # logger.debug("Decay and prune check finished.")


    def update_structure(self):
        """Updates clusters/sectors based on neuron positions using K-Means."""
        num_neurons = len(self.neurons)
        if num_neurons < PARAMETERS["N_CLUSTERS"]:
            logger.warning(f"Not enough neurons ({num_neurons}) for K-Means clustering. Skipping structure update.")
            return

        logger.info(f"Updating Nebula structure (Clustering {num_neurons} neurons)...")
        start_time = time.time()

        try:
            # 1. Get Embeddings (position, luminosity, etc.)
            neuron_ids = list(self.neurons.keys())
            embeddings = np.array([self.neurons[nid].get_embedding() for nid in neuron_ids])
            if embeddings.shape[0] != num_neurons:
                 logger.error("Mismatch between neuron count and embeddings.")
                 return
            # Check for NaNs/Infs in embeddings
            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                 logger.error("NaN or Inf detected in neuron embeddings! Cannot perform clustering.")
                 return

            # Normalize embeddings? May help K-Means
            scaler = MinMaxScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)


            # 2. Perform K-Means Clustering
            n_clusters = min(PARAMETERS["N_CLUSTERS"], num_neurons) # Ensure k <= n_samples
            kmeans = KMeans(n_clusters=n_clusters, random_state=PARAMETERS["SEED"], n_init=10) # n_init='auto' fails sometimes
            labels = kmeans.fit_predict(embeddings_scaled)

            # 3. Re-assign neurons to NEW clusters and sectors
            # Clear old structure links first
            for neuron in self.neurons.values():
                 neuron.cluster_id = None
                 neuron.sector_id = None
            old_clusters = self.clusters
            old_sectors = self.sectors
            self.clusters = {}
            self.sectors = {}
            for sector in old_sectors.values(): sector.cluster_ids.clear() # Clear refs in old sectors

            # Create new clusters based on K-Means results
            new_clusters_temp = defaultdict(list) # Temp store {label: [neuron_id]}
            for i, neuron_id in enumerate(neuron_ids):
                 new_clusters_temp[labels[i]].append(neuron_id)

            # Create Cluster objects and assign neurons
            for label, members in new_clusters_temp.items():
                 if not members: continue
                 cluster_id = self._get_new_id("cluster")
                 # Calculate cluster center (use average position of members)
                 member_positions = [self.neurons[nid].position.data for nid in members]
                 cluster_center = torch.mean(torch.stack(member_positions), dim=0)

                 cluster = Cluster(cluster_id, cluster_center, self.device)
                 self.clusters[cluster_id] = cluster
                 for nid in members:
                      cluster.add_neuron(self.neurons[nid]) # Assigns neuron.cluster_id

            # 4. Assign Clusters to Sectors (e.g., based on proximity or create new)
            # Simple approach: Re-assign each new cluster to nearest SECTOR center (or make new sectors)
            # This might be too slow, consider assigning based on cluster center directly
            temp_sector_assignment = {} # {cluster_id: sector_id}
            sector_centers = {sid: s.position for sid, s in old_sectors.items()} # Use old centers initially?

            for cluster_id, cluster in self.clusters.items():
                if not sector_centers: # No existing sectors or centers
                    sector_id = self._get_new_id("sector")
                    sector = Sector(sector_id, cluster.position.clone(), self.device)
                    self.sectors[sector_id] = sector
                    sector_centers[sector_id] = sector.position # Add new center
                    target_sector_id = sector_id
                else:
                    nearest_sector = self._find_nearest_structure(cluster.position, old_sectors) # Find nearest OLD sector center
                    if nearest_sector:
                         target_sector_id = nearest_sector.id
                         # Add cluster to this existing sector (if it still exists or we recreate it)
                         if target_sector_id not in self.sectors: # Recreate if needed
                             self.sectors[target_sector_id] = Sector(target_sector_id, nearest_sector.position.clone(), self.device)
                         self.sectors[target_sector_id].add_cluster(cluster)
                    else: # Fallback if find_nearest fails
                         sector_id = self._get_new_id("sector")
                         sector = Sector(sector_id, cluster.position.clone(), self.device)
                         self.sectors[sector_id] = sector
                         target_sector_id = sector_id

                temp_sector_assignment[cluster_id] = target_sector_id
                 # Assign sector_id to neurons in the cluster
                for nid in cluster.neuron_ids:
                    if nid in self.neurons:
                        self.neurons[nid].sector_id = target_sector_id


            # Cleanup: Remove empty sectors from the new self.sectors dict
            empty_sectors = [sid for sid, s in self.sectors.items() if not s.cluster_ids]
            for sid in empty_sectors:
                del self.sectors[sid]

            # 5. Update Positions of Clusters/Sectors to Centroids
            for cluster in self.clusters.values():
                 cluster.update_position(self.neurons)
            for sector in self.sectors.values():
                 sector.update_position(self.clusters)

            self.kdtree_needs_update = True # Positions likely changed
            elapsed = time.time() - start_time
            logger.info(f"Structure update completed in {elapsed:.4f} seconds. New structure: {len(self.sectors)} sectors, {len(self.clusters)} clusters.")

        except Exception as e:
            logger.error(f"Error during structure update: {e}", exc_info=True)
            # Avoid leaving space in inconsistent state? Maybe restore old structure?
            # For now, just log the error.


    # --- State Management ---
    def get_space_state(self) -> Dict[str, Any]:
        """Gets the serializable state of the NebulaSpace."""
        logger.debug("Getting NebulaSpace state...")
        return {
            "dimensions": convert_to_numpy(self.dimensions),
            # Serialize neurons individually
            "neurons": {nid: n.get_state() for nid, n in self.neurons.items()},
            "clusters": {cid: c.get_state() for cid, c in self.clusters.items()},
            "sectors": {sid: s.get_state() for sid, s in self.sectors.items()},
            "next_neuron_id": self.next_neuron_id,
            "next_cluster_id": self.next_cluster_id,
            "next_sector_id": self.next_sector_id,
            # Serialize graph using node_link_data for JSON compatibility if needed,
            # but pickle handles NetworkX graphs directly.
            "connection_graph": self.connection_graph # Store graph object directly for pickle
             # "connection_graph": nx.node_link_data(self.connection_graph) # For JSON
        }

    def load_space_state(self, state: Dict[str, Any]):
        """Loads the NebulaSpace state from a dictionary."""
        logger.info("Loading NebulaSpace state...")
        try:
            self.dimensions = torch.tensor(state["dimensions"], dtype=torch.float32, device=self.device)
            self.next_neuron_id = state["next_neuron_id"]
            self.next_cluster_id = state["next_cluster_id"]
            self.next_sector_id = state["next_sector_id"]

            # Clear existing state before loading
            self.neurons.clear()
            self.clusters.clear()
            self.sectors.clear()
            self.connection_graph = nx.Graph()

            # Load neurons first (they are independent nn.Modules)
            loaded_neuron_ids = set()
            if 'neurons' in state:
                for neuron_id_str, neuron_state in state["neurons"].items():
                    neuron_id = int(neuron_id_str) # Keys might be strings from JSON
                    try:
                         # Pass device explicitly to from_state
                         neuron = QuantumNeuron.from_state(neuron_state, self.device)
                         self.neurons[neuron_id] = neuron
                         loaded_neuron_ids.add(neuron_id)
                    except Exception as e:
                         logger.error(f"Failed to load neuron {neuron_id}: {e}", exc_info=False)

            # Load clusters
            loaded_cluster_ids = set()
            if 'clusters' in state:
                 for cluster_id, cluster_state in state["clusters"].items():
                      try:
                           cluster = Cluster.from_state(cluster_state, self.device)
                           self.clusters[cluster_id] = cluster
                           loaded_cluster_ids.add(cluster_id)
                      except Exception as e:
                           logger.error(f"Failed to load cluster {cluster_id}: {e}", exc_info=False)


            # Load sectors
            if 'sectors' in state:
                 for sector_id, sector_state in state["sectors"].items():
                      try:
                           sector = Sector.from_state(sector_state, self.device)
                           self.sectors[sector_id] = sector
                      except Exception as e:
                           logger.error(f"Failed to load sector {sector_id}: {e}", exc_info=False)


            # --- Rebuild relationships ---
            # Assign neurons to clusters (using loaded cluster_id from neuron state)
            for neuron_id, neuron in self.neurons.items():
                 cluster_id = neuron.cluster_id # Get ID from loaded neuron state
                 if cluster_id and cluster_id in self.clusters:
                     self.clusters[cluster_id].neuron_ids.add(neuron_id) # Add neuron ID to cluster's set
                 elif cluster_id:
                      logger.warning(f"Neuron {neuron_id} refers to missing cluster {cluster_id}. Leaving unassigned.")
                      neuron.cluster_id = None


            # Assign clusters to sectors (using loaded cluster_ids from sector state)
            for sector_id, sector in self.sectors.items():
                 # Make sure cluster IDs in sector state actually exist in loaded clusters
                 valid_cluster_ids = set(sector.cluster_ids).intersection(loaded_cluster_ids)
                 sector.cluster_ids = valid_cluster_ids
                 # Assign sector_id to neurons within the valid clusters
                 for cluster_id in valid_cluster_ids:
                      if cluster_id in self.clusters:
                           for neuron_id in self.clusters[cluster_id].neuron_ids:
                                if neuron_id in self.neurons:
                                     self.neurons[neuron_id].sector_id = sector_id


            # Load connection graph (directly if using pickle)
            if 'connection_graph' in state:
                 loaded_graph = state["connection_graph"]
                 if isinstance(loaded_graph, nx.Graph):
                      self.connection_graph = loaded_graph
                      # Prune graph nodes/edges that don't correspond to loaded neurons
                      nodes_to_remove = [node for node in self.connection_graph if node not in loaded_neuron_ids]
                      self.connection_graph.remove_nodes_from(nodes_to_remove)
                      logger.info(f"Loaded connection graph with {self.connection_graph.number_of_nodes()} nodes and {self.connection_graph.number_of_edges()} edges.")
                 # elif isinstance(loaded_graph, dict): # If using node_link_data (JSON)
                 #     self.connection_graph = nx.node_link_graph(loaded_graph)
                 #     # Prune graph nodes...
                 else:
                      logger.warning("Connection graph in saved state has unexpected format. Rebuilding from neuron connections.")
                      self.connection_graph = nx.Graph() # Reset
                      self.connection_graph.add_nodes_from(loaded_neuron_ids)
                      # Add edges based on loaded neuron.connections lists
                      for nid, neuron in self.neurons.items():
                          for target_id, strength in neuron.connections:
                              if target_id in loaded_neuron_ids and not self.connection_graph.has_edge(nid, target_id):
                                   self.connection_graph.add_edge(nid, target_id, weight=strength)


            self.kdtree_needs_update = True
            logger.info("NebulaSpace state loaded successfully.")
        except Exception as e:
            logger.critical(f"CRITICAL ERROR loading NebulaSpace state: {e}", exc_info=True)
            # Decide how to handle critical load failure: exit or re-initialize?
            logger.warning("Re-initializing NebulaSpace due to load failure.")
            self.__init__(tuple(convert_to_numpy(self.dimensions).astype(int)), self.device) # Re-init with current dims
            self.initialize_neurons(PARAMETERS["INITIAL_NEURONS"]) # Start fresh


    def initialize_neurons(self, num_neurons: int):
        """Creates and adds the initial set of neurons."""
        logger.info(f"Initializing NebulaSpace with {num_neurons} neurons...")
        added_count = 0
        for _ in range(num_neurons):
            neuron_id = self._get_new_id("neuron")
            position = torch.rand(3, device=self.device) * self.dimensions
            try:
                neuron = QuantumNeuron(
                    neuron_id=neuron_id,
                    position=position,
                    input_dim=PARAMETERS["INPUT_DIM"],
                    num_qubits=PARAMETERS["NUM_QUBITS"],
                    num_layers=PARAMETERS["QUANTUM_LAYERS"],
                    device=self.device,
                )
                # Apply initial genome factors? (Might not be needed if genome applied later)
                self.add_neuron(neuron, assign_structure=False) # Add without assigning structure yet
                added_count += 1
            except Exception as e:
                logger.error(f"Failed to initialize neuron {neuron_id}: {e}", exc_info=False)
                # Roll back ID counter if init fails?
                self.next_neuron_id -=1

        logger.info(f"Successfully added {added_count}/{num_neurons} neurons.")
        # Assign structure after all neurons are added
        logger.info("Assigning initial structure...")
        if self.neurons:
             # Initial clustering/assignment can be simple or use update_structure
             self.update_structure() # Use K-Means for initial structure
        self.kdtree_needs_update = True



# --- Holographic Memory (Associative Recall Simulation) ---
print("💾 Defining HolographicMemory...")
class HolographicMemory:
    """
    Simulates associative recall inspired by holographic principles, using embeddings.
    """
    def __init__(self, embedding_model: SentenceTransformer, dimensions: int):
        if not embedding_model or not isinstance(embedding_model, SentenceTransformer):
             raise ValueError("HolographicMemory requires a valid SentenceTransformer embedding model.")
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        # Store as {key: (embedding_vector, data)}
        self.memory: Dict[str, Tuple[np.ndarray, Any]] = {}
        # For faster search (optional, requires FAISS or similar)
        self.index = None # Placeholder for vector index
        logger.info("HolographicMemory (Associative Recall Simulation) initialized.")

    def encode(self, key: str, data: Any):
        """Stores data associated with the embedding of the key."""
        try:
            # Ensure key is a non-empty string
            if not key or not isinstance(key, str):
                 logger.warning(f"Invalid key for HolographicMemory encoding: {key}. Skipping.")
                 return

            # Ensure embedding model is available
            if not self.embedding_model:
                 logger.error("Embedding model not available for HolographicMemory encoding.")
                 return

            embedding = self.embedding_model.encode([key], convert_to_numpy=True)[0]

            # Validate embedding shape (optional, but good practice)
            if embedding.shape[0] != self.dimensions:
                 # Simple fix: Pad or truncate (might lose info)
                 new_embedding = np.zeros(self.dimensions, dtype=embedding.dtype)
                 size = min(embedding.shape[0], self.dimensions)
                 new_embedding[:size] = embedding[:size]
                 embedding = new_embedding
                 logger.warning(f"Adjusted embedding dimension for key '{key}' to {self.dimensions}.")

            self.memory[key] = (embedding, data)
            # TODO: Add to vector index if using one (e.g., FAISS)
            # if self.index: self.index.add(np.array([embedding]))

            logger.debug(f"Encoded key '{key}' into Holographic Memory.")

        except Exception as e:
            logger.error(f"Error encoding key '{key}' in HolographicMemory: {e}", exc_info=False)

    def decode(self, query_key: str, threshold: float = PARAMETERS["HOLOGRAPHIC_MEMORY_THRESHOLD"], top_k: int = 1) -> List[Tuple[str, Any, float]]:
        """Retrieves top_k data entries most similar to the query_key's embedding."""
        results = []
        if not self.memory or not query_key or not isinstance(query_key, str):
             return results
        if not self.embedding_model:
             logger.error("Embedding model not available for HolographicMemory decoding.")
             return results

        try:
            query_embedding = self.embedding_model.encode([query_key], convert_to_numpy=True)[0]
            # Adjust query embedding dimension if needed
            if query_embedding.shape[0] != self.dimensions:
                 new_embedding = np.zeros(self.dimensions, dtype=query_embedding.dtype)
                 size = min(query_embedding.shape[0], self.dimensions)
                 new_embedding[:size] = query_embedding[:size]
                 query_embedding = new_embedding

            # --- Perform Search ---
            # Option 1: Brute-force search (simple, OK for small memory)
            similarities = []
            keys = list(self.memory.keys())
            embeddings = np.array([emb for emb, data in self.memory.values()])

            if embeddings.size == 0: return results # No embeddings to search

            sim_scores = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]

            for i, score in enumerate(sim_scores):
                 if score >= threshold:
                     similarities.append((keys[i], score))

            # Sort by similarity (descending)
            similarities.sort(key=lambda item: item[1], reverse=True)

            # Get top_k results
            for key, score in similarities[:top_k]:
                 data = self.memory[key][1]
                 results.append((key, data, float(score))) # Include score in result

            # Option 2: FAISS or other vector index (much faster for large memory)
            # if self.index:
            #    distances, indices = self.index.search(np.array([query_embedding]), top_k)
            #    keys = list(self.memory.keys()) # Assuming index IDs correspond to memory keys
            #    for i, idx in enumerate(indices[0]):
            #        if distances[0][i] >= threshold: # Assuming distance metric relates to similarity threshold
            #            key = keys[idx]
            #            data = self.memory[key][1]
            #            score = distances[0][i] # Or convert distance to similarity
            #            results.append((key, data, score))

            if results:
                 logger.debug(f"Decoded query '{query_key}' - Found {len(results)} matches above threshold {threshold}.")
            # else:
            #      logger.debug(f"Decoded query '{query_key}' - No strong matches found.")

            return results

        except Exception as e:
            logger.error(f"Error decoding query '{query_key}' in HolographicMemory: {e}", exc_info=False)
            return []

    def get_state(self) -> Dict[str, Any]:
        """Returns the serializable state."""
        # Convert embeddings to lists for JSON compatibility if needed
        # Pickle handles numpy arrays directly
        return {"memory": self.memory, "dimensions": self.dimensions}

    def load_state(self, state: Dict[str, Any]):
        """Loads the state."""
        self.memory = state.get("memory", {})
        self.dimensions = state.get("dimensions", PARAMETERS["DIM"]) # Use default if missing
        # Rebuild index if using one
        # if self.index: self._rebuild_index()
        logger.info(f"HolographicMemory state loaded with {len(self.memory)} items.")


# --- Optical Processing Unit (Placeholder for Future FFT/Convolution tasks) ---
print("💡 Defining OpticalProcessingUnit (Placeholder)...")
class OpticalProcessingUnit:
    """
    Placeholder for simulating optical computations like FFTs or convolutions,
    distinct from the neuron interaction model.
    """
    def __init__(self, device=DEVICE):
        self.device = device
        logger.info("OpticalProcessingUnit (Placeholder) initialized.")

    def perform_fft(self, data: Union[np.ndarray, torch.Tensor]) -> Optional[torch.Tensor]:
        """Simulates an optical Fourier Transform using torch.fft."""
        if isinstance(data, np.ndarray):
             tensor = torch.from_numpy(data).to(self.device)
        elif isinstance(data, torch.Tensor):
             tensor = data.to(self.device)
        else:
             logger.error(f"Optical FFT requires NumPy array or Torch tensor, got {type(data)}")
             return None

        # Ensure complex type for FFT
        if not torch.is_complex(tensor):
             tensor = tensor.to(dtype=torch.complex64)

        try:
             if tensor.ndim == 1:
                 fft_result = torch.fft.fft(tensor)
             elif tensor.ndim == 2:
                 fft_result = torch.fft.fft2(tensor)
             elif tensor.ndim == 3:
                  fft_result = torch.fft.fftn(tensor) # For 3D data
             else:
                 logger.warning(f"FFT for {tensor.ndim}-D data not explicitly handled. Using fftn.")
                 fft_result = torch.fft.fftn(tensor)

             logger.debug("Performed simulated Optical FFT.")
             return fft_result
        except Exception as e:
             logger.error(f"Error during simulated FFT: {e}", exc_info=False)
             return None

    def perform_convolution(self, data: torch.Tensor, kernel: torch.Tensor) -> Optional[torch.Tensor]:
        """Simulates optical convolution using FFT."""
        # Implementation would involve FFT of data and kernel, multiplication in Fourier space,
        # and inverse FFT. Requires careful handling of padding and dimensions.
        logger.warning("Simulated optical convolution is not fully implemented yet.")
        # Placeholder implementation (simple PyTorch conv)
        try:
             data = data.to(self.device).float()
             kernel = kernel.to(self.device).float()
             # Adjust dimensions if necessary for torch.convNd
             # Example for 2D convolution:
             if data.ndim == 2 and kernel.ndim == 2:
                  # Add batch and channel dimensions: (N, C_in, H, W)
                  data = data.unsqueeze(0).unsqueeze(0)
                  # Kernel needs (C_out, C_in/groups, kH, kW)
                  kernel = kernel.unsqueeze(0).unsqueeze(0)
                  padding = (kernel.shape[-2] // 2, kernel.shape[-1] // 2)
                  result = F.conv2d(data, kernel, padding=padding)
                  return result.squeeze() # Remove batch/channel dims
             else:
                  logger.error("Placeholder convolution only supports 2D data/kernel.")
                  return None

        except Exception as e:
             logger.error(f"Error during placeholder convolution: {e}", exc_info=False)
             return None

# --- Knowledge Graph ---
print("🌍 Defining EnhancedKnowledgeGraph...")
class EnhancedKnowledgeGraph:
    """Manages the knowledge graph using NetworkX and Sentence Embeddings."""
    def __init__(self, embedding_model=None, graph_file=PARAMETERS["KNOWLEDGE_GRAPH_FILE"]):
        self.graph = nx.DiGraph() # Directed graph
        self.graph_file = Path(graph_file)
        # Load embedding model lazily or pass externally
        self.embedding_model = embedding_model
        self._load_embedding_model() # Attempt to load if not provided
        self.load_graph()
        logger.info(f"Knowledge Graph initialized. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")

    def _load_embedding_model(self):
        """Loads the embedding model if not already loaded."""
        if self.embedding_model is None and TRANSFORMERS_AVAILABLE:
             try:
                 logger.info(f"Loading embedding model for KG: {PARAMETERS['EMBEDDING_MODEL_NAME']}")
                 # Use SentenceTransformer directly
                 self.embedding_model = SentenceTransformer(PARAMETERS['EMBEDDING_MODEL_NAME'], device=DEVICE)
             except Exception as e:
                 logger.error(f"Failed to load embedding model for Knowledge Graph: {e}")
                 self.embedding_model = None # Ensure it's None if loading fails
        elif not TRANSFORMERS_AVAILABLE:
             logger.warning("Transformers library not available, KG embedding features disabled.")
             self.embedding_model = None


    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generates an embedding for a text node."""
        if self.embedding_model and text:
            try:
                return self.embedding_model.encode([text], convert_to_numpy=True)[0]
            except Exception as e:
                logger.error(f"Failed to generate embedding for text '{text[:50]}...': {e}")
                return None
        return None

    def add_concept(self, concept: str, attributes: Optional[Dict] = None):
        """Adds a concept node to the graph."""
        if not concept: return
        concept_key = concept.lower().strip()
        if concept_key not in self.graph:
            embedding = self._get_embedding(concept)
            node_attrs = attributes or {}
            if embedding is not None:
                node_attrs['embedding'] = embedding # Store embedding with node
            self.graph.add_node(concept_key, **node_attrs)
            logger.debug(f"KG: Added concept '{concept_key}'")

    def add_relation(self, source: str, target: str, relation: str, attributes: Optional[Dict] = None):
        """Adds a directed relation (edge) between two concepts."""
        if not source or not target or not relation: return
        source_key = source.lower().strip()
        target_key = target.lower().strip()
        # Ensure nodes exist
        self.add_concept(source_key)
        self.add_concept(target_key)
        # Add edge
        edge_attrs = attributes or {}
        edge_attrs['relation'] = relation.lower().strip()
        # Optionally add embedding for the relation phrase itself?
        # rel_embedding = self._get_embedding(relation)
        # if rel_embedding is not None: edge_attrs['embedding'] = rel_embedding
        self.graph.add_edge(source_key, target_key, **edge_attrs)
        logger.debug(f"KG: Added relation '{source_key}' -[{edge_attrs['relation']}]-> '{target_key}'")

    def get_related_concepts(self, concept: str, max_distance: int = 1) -> List[str]:
        """Finds concepts related within a certain distance."""
        concept_key = concept.lower().strip()
        if concept_key not in self.graph: return []
        # Use breadth-first search to find neighbors within distance
        related = set()
        queue = deque([(concept_key, 0)])
        visited = {concept_key}
        while queue:
             curr_node, dist = queue.popleft()
             if dist >= max_distance: continue
             # Add successors and predecessors
             for neighbor in list(self.graph.successors(curr_node)) + list(self.graph.predecessors(curr_node)):
                 if neighbor not in visited:
                     visited.add(neighbor)
                     related.add(neighbor)
                     queue.append((neighbor, dist + 1))
        return list(related)

    def find_similar_concepts(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Finds concepts with embeddings most similar to the query."""
        if not self.embedding_model:
             logger.warning("Cannot find similar concepts without embedding model.")
             return []
        query_embedding = self._get_embedding(query)
        if query_embedding is None: return []

        similarities = []
        for node, attrs in self.graph.nodes(data=True):
             if 'embedding' in attrs and attrs['embedding'] is not None:
                 node_embedding = attrs['embedding']
                 # Ensure dimensions match before calculating similarity
                 if node_embedding.shape == query_embedding.shape:
                      sim = calculate_similarity(query_embedding, node_embedding)
                      similarities.append((node, sim))

        similarities.sort(key=lambda item: item[1], reverse=True)
        return similarities[:top_k]

    def reason(self, query: str) -> List[str]:
        """Performs simple reasoning based on query structure or similarity."""
        # Basic reasoning: Find similar concepts and list their direct relations.
        # Could be expanded significantly (e.g., path finding, pattern matching).
        results = []
        similar_concepts = self.find_similar_concepts(query, top_k=1)
        if not similar_concepts:
             # Maybe try direct keyword matching if no embedding match
             keywords = query.lower().split()
             matched_nodes = [n for n in self.graph.nodes() if any(k in n for k in keywords)]
             if not matched_nodes: return ["No relevant concepts found in knowledge graph."]
             concepts_to_explore = matched_nodes[:1] # Take first match
        else:
             concepts_to_explore = [similar_concepts[0][0]] # Explore most similar

        for concept in concepts_to_explore:
             results.append(f"Found related concept: '{concept}'.")
             # List outgoing relations
             for source, target, attrs in self.graph.out_edges(concept, data=True):
                 relation = attrs.get('relation', 'related to')
                 results.append(f" - '{source}' -> '{relation}' -> '{target}'")
             # List incoming relations
             for source, target, attrs in self.graph.in_edges(concept, data=True):
                 relation = attrs.get('relation', 'related to')
                 results.append(f" - '{source}' -> '{relation}' -> '{target}'") # source leads TO concept

        return results if results else ["Could not derive information from knowledge graph."]


    def save_graph(self):
        """Saves the graph to a file (GraphML format)."""
        logger.info(f"Saving knowledge graph to {self.graph_file}...")
        try:
            # Need to convert numpy embeddings to lists for GraphML export
            graph_copy = self.graph.copy()
            for node, data in graph_copy.nodes(data=True):
                if 'embedding' in data and isinstance(data['embedding'], np.ndarray):
                    data['embedding'] = data['embedding'].tolist()
            for u, v, data in graph_copy.edges(data=True):
                 if 'embedding' in data and isinstance(data['embedding'], np.ndarray):
                     data['embedding'] = data['embedding'].tolist()

            nx.write_graphml(graph_copy, self.graph_file)
            logger.info("Knowledge graph saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")

    def load_graph(self):
        """Loads the graph from a file."""
        if self.graph_file.exists():
            logger.info(f"Loading knowledge graph from {self.graph_file}...")
            try:
                self.graph = nx.read_graphml(self.graph_file)
                # Convert loaded lists back to numpy embeddings
                for node, data in self.graph.nodes(data=True):
                    if 'embedding' in data and isinstance(data['embedding'], list):
                        data['embedding'] = np.array(data['embedding'])
                for u, v, data in self.graph.edges(data=True):
                    if 'embedding' in data and isinstance(data['embedding'], list):
                        data['embedding'] = np.array(data['embedding'])

                logger.info(f"Knowledge graph loaded. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
            except Exception as e:
                logger.error(f"Failed to load knowledge graph: {e}. Starting with an empty graph.")
                self.graph = nx.DiGraph()
        else:
            logger.info("Knowledge graph file not found. Starting with an empty graph.")
            self.graph = nx.DiGraph()

    def get_state(self) -> Dict[str, Any]:
         """Returns graph state (delegates to saving mechanism)."""
         # For pickle, just return the graph object
         return {"graph": self.graph}

    def load_state(self, state: Dict[str, Any]):
         """Loads graph state."""
         if "graph" in state and isinstance(state["graph"], nx.DiGraph):
              self.graph = state["graph"]
              logger.info("Knowledge graph state loaded from state object.")
         else:
              logger.warning("Could not load knowledge graph from state object. Loading from file or starting empty.")
              self.load_graph() # Fallback to file


# --- Evolutionary Components ---
print("🧬 Defining Evolutionary Components...")
class NebulaGenome:
    """Represents the genetic parameters of the Nebula system."""
    # Define parameters that can be evolved
    parameter_ranges = {
        "connection_probability": (0.02, 0.25),
        "max_connections": (5, 25),
        "light_attenuation_factor": (0.1, 0.8),
        "light_receive_factor": (0.01, 0.1),
        "luminosity_decay": (0.001, 0.02),
        "structure_update_interval": (10, 100),
        "neuron_activity_threshold": (0.05, 0.3),
        # Add more tunable PARAMETERS keys here if desired
        # e.g., "POSITION_UPDATE_DT": (0.05, 0.5),
    }
    int_params = {"max_connections", "structure_update_interval"} # Params requiring integer values

    def __init__(self):
        """Initializes with random values within ranges."""
        for param, (min_val, max_val) in self.parameter_ranges.items():
            if param in self.int_params:
                setattr(self, param, random.randint(min_val, max_val))
            else:
                setattr(self, param, random.uniform(min_val, max_val))

    def to_list(self) -> List[float]:
        """Converts genome to a list for DEAP."""
        return [getattr(self, param) for param in self.parameter_ranges.keys()]

    @classmethod
    def from_list(cls, data: List[float]) -> "NebulaGenome":
        """Creates a genome from a list."""
        genome = cls()
        for i, param in enumerate(cls.parameter_ranges.keys()):
            value = data[i]
            if param in cls.int_params:
                value = int(round(value)) # Ensure integer value
            # Clamp value within defined ranges during creation from list
            min_val, max_val = cls.parameter_ranges[param]
            value = max(min_val, min(max_val, value))
            setattr(genome, param, value)
        return genome

    def apply_to_parameters(self):
        """Updates the global PARAMETERS dictionary with genome values."""
        logger.debug("Applying genome values to global PARAMETERS...")
        for param in self.parameter_ranges.keys():
            value = getattr(self, param)
            # Convert parameter name if needed (e.g., genome name vs PARAMETERS key)
            param_key = param.upper() # Assuming PARAMETERS keys are uppercase versions
            if param_key in PARAMETERS:
                PARAMETERS[param_key] = value
                # logger.debug(f"  Set PARAMETERS['{param_key}'] = {value}")
            else:
                logger.warning(f"Genome parameter '{param}' does not match any key in PARAMETERS (tried '{param_key}').")


# --- Self-Correction Components (Placeholders/Basic Structure) ---
print("🔧 Defining Self-Correction Components...")

class ErrorDetector:
    """Detects errors from logs or runtime exceptions."""
    def __init__(self):
        self.error_patterns = [
            re.compile(r".*Traceback \(most recent call last\):.*", re.DOTALL),
            re.compile(r".*(Error|Exception):.*"),
            # Add more specific patterns if needed
        ]

    def detect(self, log_entry: str = None, exception: Exception = None) -> Optional[Dict]:
        """Detects errors from log entry or exception object."""
        if exception:
            error_type = type(exception).__name__
            error_msg = str(exception)
            tb_list = traceback.format_exception(type(exception), exception, exception.__traceback__)
            trace = "".join(tb_list)
            logger.debug(f"Detected error via exception: {error_type}: {error_msg}")
            return {"type": error_type, "message": error_msg, "traceback": trace}

        if log_entry:
            for pattern in self.error_patterns:
                match = pattern.search(log_entry)
                if match:
                    # Try to extract basic info (very naive)
                    error_type = "UnknownError"
                    error_msg = log_entry[:200] # Take first 200 chars as message
                    if "Error:" in log_entry:
                         error_type = log_entry.split("Error:", 1)[0].split()[-1] + "Error"
                         error_msg = log_entry.split("Error:", 1)[1].strip()
                    elif "Exception:" in log_entry:
                         error_type = log_entry.split("Exception:", 1)[0].split()[-1] + "Exception"
                         error_msg = log_entry.split("Exception:", 1)[1].strip()

                    logger.debug(f"Detected error via log pattern: {error_type}")
                    return {"type": error_type, "message": error_msg, "traceback": log_entry} # Use full log as trace
        return None

class CodeTester:
    """(Simplified) Tests code snippets in a restricted environment."""
    def test_snippet(self, code_snippet: str, context: Dict = None) -> Tuple[bool, str]:
        """
        Executes a snippet and checks for exceptions.
        WARNING: This is a simplified version. Real-world safe execution is complex.
                 Using exec() is inherently risky. A sandboxed environment is needed
                 for true safety, which is beyond this scope.
        """
        logger.warning("Executing CodeTester.test_snippet using exec(). THIS IS INSECURE for untrusted code.")
        try:
            # Create a restricted global/local scope for exec
            restricted_globals = {"__builtins__": {
                 "print": print, "range": range, "len": len, "list": list, "dict": dict, "str": str, "int": int, "float": float, "True": True, "False": False, "None": None,
                 "Exception": Exception, # Allow basic exception handling
                 "math": math, # Allow math module
                 "torch": torch, # Allow torch (potentially risky)
                 "np": np, # Allow numpy
                 }}
            restricted_locals = context or {}
            # Execute the snippet
            exec(code_snippet, restricted_globals, restricted_locals)
            logger.debug("Code snippet executed without raising exceptions.")
            return True, "Execution successful (no exceptions)."
        except Exception as e:
            logger.warning(f"Code snippet failed execution: {e}", exc_info=False)
            return False, f"Execution failed: {type(e).__name__}: {e}"

class ErrorCorrector:
    """(Conceptual) Uses LLMs to suggest corrections for code errors."""
    def __init__(self, nebula_code: str, codegen_pipeline=None):
        self.nebula_code = nebula_code # Store the full code text
        # Requires a code generation model/pipeline
        if codegen_pipeline is None and TRANSFORMERS_AVAILABLE:
            logger.warning("ErrorCorrector initialized without a CodeGen pipeline. Correction features disabled.")
            self.codegen_pipeline = None
        else:
             self.codegen_pipeline = codegen_pipeline


    @require_component('codegen_pipeline')
    def suggest_correction(self, error_info: Dict, code_context: str) -> Optional[str]:
        """Suggests a code correction using an LLM."""
        if not error_info or not code_context: return None

        prompt = f"""The following Python code produced an error:
--- ERROR ---
Type: {error_info.get('type', 'Unknown')}
Message: {error_info.get('message', 'N/A')}
Traceback (partial):
{error_info.get('traceback', 'N/A')[-1000:]} # Limit traceback length

--- CODE CONTEXT ---
{code_context}

--- TASK ---
Analyze the error and the code context. Provide a corrected version of the relevant lines from the CODE CONTEXT only. Output ONLY the corrected Python code snippet. Do not include explanations or surrounding text.
Corrected Code Snippet:
"""
        try:
            logger.debug("Querying CodeGen LLM for error correction...")
            # Use the pipeline directly
            results = self.codegen_pipeline(prompt, max_length=len(prompt.split()) + 200, num_return_sequences=1, temperature=0.3, truncation=True) # Adjust max_length
            if results and isinstance(results, list):
                generated_text = results[0]['generated_text']
                # Extract the code part after the prompt
                suggestion = generated_text.split("Corrected Code Snippet:", 1)[-1].strip()
                # Basic cleanup (remove ```python, ``` etc.)
                suggestion = re.sub(r"```python\n?|```", "", suggestion).strip()
                logger.info(f"CodeGen LLM suggested correction:\n{suggestion}")
                # Basic validation: Is it non-empty? Does it look like Python?
                if suggestion and len(suggestion) > 1: # and ('def ' in suggestion or 'class ' in suggestion or '=' in suggestion):
                    return suggestion
                else:
                     logger.warning("CodeGen LLM suggestion seems invalid or empty.")
                     return None
            else:
                 logger.warning("CodeGen LLM returned unexpected result format.")
                 return None

        except Exception as e:
            logger.error(f"Error querying CodeGen LLM for correction: {e}", exc_info=False)
            return None

class LearningSystem:
    """(Conceptual) Learns from successful corrections or analyzes code."""
    def learn_from_correction(self, original_code: str, corrected_code: str, error_info: Dict):
        """Stores information about successful corrections."""
        # Placeholder: Could add this pair to a fine-tuning dataset for the CodeGen model
        # Or add patterns to the Knowledge Graph
        logger.info("Learning from successful correction (conceptual).")
        pass

    def analyze_code_quality(self, code_snippet: str) -> float:
         """Analyzes code quality (placeholder)."""
         # Placeholder: Could use static analysis tools, linters, or another LLM
         # Simple heuristic: length, presence of comments, basic structure
         score = 0.5 # Default score
         if code_snippet:
              lines = code_snippet.strip().split('\n')
              num_lines = len(lines)
              num_comments = sum(1 for line in lines if line.strip().startswith('#'))
              score += 0.1 * min(1.0, num_lines / 50.0) # Bonus for length (up to 50 lines)
              score += 0.2 * min(1.0, num_comments / (num_lines + 1e-6)) # Bonus for comments
         return np.clip(score, 0.1, 0.9).item()


class NebulaErrorCorrection:
    """Orchestrates the error detection, correction, and learning process."""
    def __init__(self, error_detector: ErrorDetector, error_corrector: ErrorCorrector,
                 learning_system: LearningSystem, code_tester: CodeTester, nebula: 'NebulaAGI'):
        self.detector = error_detector
        self.corrector = error_corrector
        self.learner = learning_system
        self.tester = code_tester
        self.nebula = nebula # Reference to the main NebulaAGI instance
        logger.info("NebulaErrorCorrection system initialized.")

    def handle_runtime_error(self, exception: Exception, code_context: Optional[str] = None):
        """Handles a detected runtime error."""
        if not PARAMETERS["SELF_CORRECTION_ENABLED"]:
             logger.warning("Self-correction disabled. Skipping handling.")
             return False # Indicate correction was not attempted/applied

        logger.warning(f"Attempting self-correction for error: {exception}")
        error_info = self.detector.detect(exception=exception)
        if not error_info:
            logger.error("Could not extract error info from exception.")
            return False

        if not code_context:
             # Try to get context from the traceback if not provided
             tb = error_info.get('traceback', '')
             # Find the last file reference in the traceback related to Nebula's code
             match = re.search(r'File "(.*?)", line (\d+), in (\S+)', tb)
             if match:
                 file_path, line_num, func_name = match.groups()
                 line_num = int(line_num)
                 # Be careful only to get context for Nebula's own code file
                 if Path(file_path).name == Path(__file__).name: # Check if error is in this file
                     try:
                         full_code = self.nebula.get_full_code_text()
                         lines = full_code.splitlines()
                         start_line = max(0, line_num - 10)
                         end_line = min(len(lines), line_num + 10)
                         code_context = "\n".join(lines[start_line:end_line])
                         logger.debug(f"Extracted code context around line {line_num}")
                     except Exception as ctx_e:
                          logger.error(f"Failed to extract code context automatically: {ctx_e}")
                          code_context = "Context unavailable."
                 else:
                      code_context = "Error occurred outside NebulaAGI main script."
             else:
                 code_context = "Could not automatically determine code context."


        # Suggest correction using LLM
        if not self.corrector or not self.corrector.codegen_pipeline:
             logger.error("ErrorCorrector or its CodeGen pipeline not available. Cannot suggest correction.")
             return False

        suggested_correction = self.corrector.suggest_correction(error_info, code_context)
        if not suggested_correction:
            logger.error("Failed to get correction suggestion from LLM.")
            return False

        # Test the correction (simplified)
        test_passed, test_output = self.tester.test_snippet(suggested_correction)
        logger.info(f"Correction test result: Passed={test_passed}, Output: {test_output}")

        if test_passed:
             # --- Apply the correction ---
             # This is the most dangerous part. Requires replacing code in the running file.
             logger.warning(f"Attempting to apply correction (EXPERIMENTAL & RISKY):\n{suggested_correction}")
             apply_success = self.nebula.apply_code_modification(code_context, suggested_correction)

             if apply_success:
                 logger.info("✅ Correction applied successfully (Code modified).")
                 # Learn from the successful correction
                 self.learner.learn_from_correction(code_context, suggested_correction, error_info)
                 return True # Indicate correction was applied
             else:
                 logger.error("❌ Failed to apply the tested correction to the source file.")
                 return False
        else:
             logger.warning("Suggested correction failed testing. Discarding.")
             return False

# ========================================================
# NEBULA AGI - MAIN CLASS
# ========================================================
print("✨ Defining NebulaAGI Main Class...")

# --- Forward declaration for type hinting ---
class UserInterface: pass
class NebulaThread: pass

class NebulaAGI:
    """
    The unified Nebula AGI simulation framework core.
    """
    VERSION = "1.5"

    def __init__(self):
        """Initializes the Nebula AGI system."""
        self._print_banner()
        self.initialized = False
        self.start_time = time.time()
        self.iteration = 0
        self.last_backup_time = 0
        self.last_evolution_time = 0
        self.last_improvement_check_time = 0
        self.last_structure_update_time = 0
        self.shutdown_requested = False

        # --- History & Monitoring ---
        self.error_history = deque(maxlen=PARAMETERS["MAX_ERROR_HISTORY"])
        self.modification_history = deque(maxlen=PARAMETERS["MAX_MODIFICATION_HISTORY"])
        self.performance_history = deque(maxlen=200) # Track evaluation scores
        self.llm_interaction_log = deque(maxlen=100) # Log LLM prompts/responses

        # --- Core Components (Initialize with placeholders first) ---
        self.device = DEVICE
        self.space: Optional[NebulaSpace] = None
        self.knowledge_graph: Optional[EnhancedKnowledgeGraph] = None
        self.holographic_memory: Optional[HolographicMemory] = None
        self.optical_processor: Optional[OpticalProcessingUnit] = None
        self.genome: Optional[NebulaGenome] = None
        self.genetic_algorithm: Optional[base.Toolbox] = None
        self.error_correction_system: Optional[NebulaErrorCorrection] = None
        self.llm_professor: Optional[Any] = None # Placeholder for LLMProfessor class if defined
        self.image_analyzer: Optional[Any] = None # Placeholder for ImageAnalyzer class if defined
        self.code_analyzer: Optional[Any] = None # Placeholder for CodeSearchAndAnalyzer class if defined

        # NLP Tools
        self.spacy_nlp = None
        self._load_nlp_tools() # Load SpaCy early

        # --- LLM Management ---
        self.llm_models: Dict[str, Optional[Dict[str, Any]]] = {}
        self.llm_load_status: Dict[str, bool] = {}
        if TRANSFORMERS_AVAILABLE:
            self._init_llm_placeholders()
        else:
             logger.warning("Transformers library not available. LLM features disabled.")


        # --- Component Initialization Sequence ---
        init_steps = [
            ("NebulaSpace", self._init_neural_space),
            ("Knowledge Systems", self._init_knowledge_systems), # Requires embedding LLM
            ("Evolution Engine", self._init_evolution_engine),
            ("Processing Units", self._init_processing_units),
            ("Analysis Tools", self._init_analysis_tools),
            ("Error Correction", self._init_error_correction), # Requires other components & LLMs
        ]

        initialization_successful = True
        for name, init_func in init_steps:
            logger.info(f"Initializing: {name}...")
            try:
                if not init_func():
                    logger.error(f"❌ Failed to initialize {name}. System may be unstable.")
                    # Decide if failure is critical. For now, continue but log error.
                    # initialization_successful = False
            except Exception as e:
                logger.critical(f"💥 CRITICAL ERROR during initialization of {name}: {e}", exc_info=True)
                initialization_successful = False
                break # Stop initialization on critical error

        if not initialization_successful:
             logger.critical("Nebula AGI initialization failed. Exiting.")
             sys.exit(1)

        # Load previous state AFTER components are initialized (if file exists)
        self.load_state()

        # Apply the current genome to parameters after potential state loading
        if self.genome:
            self.genome.apply_to_parameters()

        # --- UI Initialization (if enabled) ---
        self.app: Optional[QApplication] = None
        self.user_interface: Optional[UserInterface] = None
        self.nebula_thread: Optional[NebulaThread] = None
        if PARAMETERS["UI_ENABLED"]:
            self._init_ui()

        self.initialized = True
        logger.info("✅ NEBULA AGI Initialization Complete.")
        self.display_statistics()

    def _print_banner(self):
        """Prints a startup banner."""
        print("*" * 70)
        print(f"🚀 Initializing NEBULA AGI Simulation Framework v{self.VERSION} 🚀")
        print("*" * 70)
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python Version: {sys.version.split()[0]}")
        print(f"Torch Version: {torch.__version__}")
        print(f"PennyLane Version: {qml.__version__}")
        if TRANSFORMERS_AVAILABLE: print(f"Transformers Version: {importlib.metadata.version('transformers')}")
        if DEAP_AVAILABLE: print(f"DEAP Version: {importlib.metadata.version('deap')}")
        print("-" * 70)


    def _load_nlp_tools(self):
        """Loads spaCy NLP model."""
        if not NLP_AVAILABLE:
            logger.warning("SpaCy not available. Skipping NLP tool loading.")
            return
        try:
            logger.info("Loading spaCy NLP model (en_core_web_sm)...")
            self.spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded.")
        except OSError:
            logger.warning("spaCy 'en_core_web_sm' not found. Attempting download...")
            try:
                spacy.cli.download("en_core_web_sm")
                self.spacy_nlp = spacy.load("en_core_web_sm")
                logger.info("SpaCy model downloaded and loaded.")
            except Exception as e:
                logger.error(f"Failed to download or load spaCy model: {e}. NLP features limited.")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")


    def _init_llm_placeholders(self):
        """Sets up the structure for managing LLMs."""
        self.llm_models = {
            "embedding": {"config_key": "EMBEDDING_MODEL_NAME", "model": None, "tokenizer": None, "pipeline": None, "processor": None, "last_used": 0},
            "text_generation_small": {"config_key": "GENERATION_MODEL_SMALL", "model": None, "tokenizer": None, "pipeline": None, "processor": None, "last_used": 0},
            "text_generation_large": {"config_key": "GENERATION_MODEL_LARGE", "model": None, "tokenizer": None, "pipeline": None, "processor": None, "last_used": 0},
            "code_generation": {"config_key": "CODEGEN_MODEL_NAME", "model": None, "tokenizer": None, "pipeline": None, "processor": None, "last_used": 0},
            "image_captioning": {"config_key": "IMAGE_CAPTION_MODEL_NAME", "model": None, "tokenizer": None, "pipeline": None, "processor": None, "last_used": 0},
            "qa": {"config_key": "QA_MODEL_NAME", "model": None, "tokenizer": None, "pipeline": None, "processor": None, "last_used": 0},
        }
        self.llm_load_status = {k: False for k in self.llm_models}


    def _init_neural_space(self) -> bool:
        """Initializes the NebulaSpace."""
        try:
            self.space = NebulaSpace(
                dimensions=PARAMETERS["SPACE_DIMENSIONS"],
                device=self.device
            )
            # Neurons are added either by loading state or explicitly calling initialize_neurons
            if not PARAMETERS["STATE_FILE"].exists(): # Initialize only if no state file
                 self.space.initialize_neurons(PARAMETERS["INITIAL_NEURONS"])
            return True
        except Exception as e:
            logger.error(f"Failed to initialize NebulaSpace: {e}", exc_info=True)
            return False

    def _init_knowledge_systems(self) -> bool:
        """Initializes Knowledge Graph and Holographic Memory."""
        # Requires embedding model
        if not self.is_llm_loaded("embedding"):
             if not self.load_llm_model("embedding"):
                  logger.error("Failed to load embedding model. Cannot initialize knowledge systems.")
                  return False

        embedding_model_instance = self.llm_models["embedding"]["model"] if self.llm_models.get("embedding") else None

        if not embedding_model_instance or not isinstance(embedding_model_instance, SentenceTransformer):
             logger.error(f"Embedding model instance is invalid or not SentenceTransformer: {type(embedding_model_instance)}")
             return False

        embedding_dim = embedding_model_instance.get_sentence_embedding_dimension()
        if not embedding_dim:
             logger.error("Could not determine embedding dimension from SentenceTransformer model.")
             return False

        try:
             # Pass the loaded SentenceTransformer instance
             self.knowledge_graph = EnhancedKnowledgeGraph(embedding_model=embedding_model_instance)
             self.holographic_memory = HolographicMemory(embedding_model=embedding_model_instance, dimensions=embedding_dim)
             return True
        except Exception as e:
             logger.error(f"Failed to initialize knowledge systems: {e}", exc_info=True)
             return False

    def _init_evolution_engine(self) -> bool:
        """Initializes the Genetic Algorithm engine."""
        if not PARAMETERS["EVOLUTION_ENABLED"]:
             logger.warning("Evolutionary features disabled (DEAP not found or disabled in params).")
             return True # Not a critical failure

        try:
             self.genome = NebulaGenome() # Initialize with default random values
             self.genetic_algorithm = self._setup_genetic_algorithm()
             return True
        except Exception as e:
             logger.error(f"Failed to initialize evolution engine: {e}", exc_info=True)
             return False

    def _init_processing_units(self) -> bool:
        """Initializes conceptual processing units."""
        try:
            self.optical_processor = OpticalProcessingUnit(device=self.device)
            # Add other conceptual units here if needed
            return True
        except Exception as e:
             logger.error(f"Failed to initialize processing units: {e}", exc_info=True)
             return False

    def _init_analysis_tools(self) -> bool:
        """Initializes analysis tools (image, code etc.)."""
        # These might depend on LLMs being loadable
        # Initialize placeholders, actual functionality might require loading models later
        # Example:
        # self.image_analyzer = DimensionSafeImageAnalyzer() # Define this class if needed
        # self.code_analyzer = CodeSearchAndAnalyzer()      # Define this class if needed
        logger.info("Analysis tools initialized (placeholders).")
        return True

    def _init_error_correction(self) -> bool:
        """Initializes the error correction system."""
        if not PARAMETERS["SELF_CORRECTION_ENABLED"]:
             logger.warning("Self-correction features disabled.")
             return True # Not a failure if disabled

        # Requires several components, including a CodeGen LLM
        if not self.is_llm_loaded("code_generation"):
             if not self.load_llm_model("code_generation"):
                  logger.error("Failed to load CodeGen LLM. Cannot initialize full Error Correction system.")
                  # Allow basic detection to work without correction?
                  # return False # Mark as failure if full system needed
                  self.error_correction_system = None
                  return True # Allow continuation without correction capability

        codegen_pipeline_instance = self.get_llm_pipeline("code_generation")
        if not codegen_pipeline_instance:
             logger.error("CodeGen pipeline not available after loading model. Cannot initialize Error Corrector.")
             self.error_correction_system = None
             return True # Continue without correction

        try:
            error_detector = ErrorDetector()
            code_tester = CodeTester() # Simplified tester
            # Get current code text for corrector context
            try:
                 nebula_code = self.get_full_code_text()
            except Exception as code_e:
                 logger.error(f"Failed to read Nebula source code for ErrorCorrector: {code_e}")
                 nebula_code = "# Could not read source code."

            error_corrector = ErrorCorrector(nebula_code, codegen_pipeline_instance)
            learning_system = LearningSystem() # Conceptual learner

            self.error_correction_system = NebulaErrorCorrection(
                 error_detector=error_detector,
                 error_corrector=error_corrector,
                 learning_system=learning_system,
                 code_tester=code_tester,
                 nebula=self
            )
            return True
        except Exception as e:
             logger.error(f"Failed to initialize error correction system: {e}", exc_info=True)
             self.error_correction_system = None
             return False

    def _init_ui(self):
        """Initializes the PyQt UI if enabled and available."""
        if not PYQT_AVAILABLE:
            logger.warning("PyQt6 not found. UI disabled.")
            return False
        try:
             logger.info("Initializing User Interface...")
             self.app = QApplication.instance() or QApplication(sys.argv)
             # Dynamically import UI classes only if needed and available
             from nebula_ui import UserInterface, NebulaThread # Assume UI code is in nebula_ui.py
             self.user_interface = UserInterface(self)
             self.nebula_thread = NebulaThread(self)
             # Connect signals
             self.nebula_thread.log_signal.connect(self.user_interface.log_message)
             self.nebula_thread.response_signal.connect(self.user_interface.display_message)
             self.nebula_thread.stats_signal.connect(self.user_interface.update_stats) # Add signal for stats
             self.user_interface.show()
             logger.info("UI Initialized and displayed.")
             return True
        except ImportError:
             logger.error("Failed to import UI components from nebula_ui.py. UI disabled.")
             PARAMETERS["UI_ENABLED"] = False
             return False
        except Exception as e:
             logger.error(f"Failed to initialize UI: {e}", exc_info=True)
             PARAMETERS["UI_ENABLED"] = False
             return False


    # --- LLM Loading & Management ---

    def is_llm_loaded(self, model_key: str) -> bool:
        """Checks if a specific LLM is loaded."""
        return self.llm_load_status.get(model_key, False)

    def load_llm_model(self, model_key: str) -> bool:
        """Loads the specified LLM model, tokenizer/processor, and pipeline."""
        if not TRANSFORMERS_AVAILABLE:
             logger.error("Transformers library not available. Cannot load LLMs.")
             return False
        if model_key not in self.llm_models:
            logger.error(f"LLM key '{model_key}' not found in configuration.")
            return False
        if self.is_llm_loaded(model_key):
            # Update last used time
            self.llm_models[model_key]["last_used"] = time.time()
            return True

        # Check memory pressure before loading
        if self._check_memory_usage(f"before loading {model_key}", trigger_unload=False): # Just check, don't unload yet
             logger.warning(f"High memory pressure detected before loading {model_key}. Loading may fail.")

        config_key = self.llm_models[model_key]["config_key"]
        model_name = PARAMETERS.get(config_key)
        if not model_name:
            logger.error(f"Model name not found in PARAMETERS for config key: {config_key}")
            return False

        logger.info(f"Loading LLM '{model_key}' ({model_name})...")
        load_start_time = time.time()

        # --- Determine Model Type and Task ---
        model_type = "text" # Default
        task = None
        trust_remote_code = False
        quantization_config = None # Example: BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = torch.float32 # Default

        if model_key == "embedding":
            model_type = "embedding"
        elif model_key == "text_generation_large":
             task = "text-generation"
             trust_remote_code = PARAMETERS.get("TRUST_REMOTE_CODE_LARGE_GEN", False)
             # Use bfloat16 if available on GPU for faster computation/less memory
             if DEVICE.type == 'cuda' and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                  torch_dtype = torch.bfloat16
                  logger.info("Using bfloat16 for large generation model.")
        elif model_key == "text_generation_small":
            task = "text-generation"
        elif model_key == "code_generation":
            task = "text-generation"
        elif model_key == "image_captioning":
            model_type = "multimodal"
            task = "image-to-text"
        elif model_key == "qa":
            task = "question-answering"

        # --- Load Model and Tokenizer/Processor ---
        model = None
        tokenizer = None
        processor = None
        load_success = False

        try:
            load_options = {"cache_dir": PARAMETERS["MODEL_CACHE_DIR"]}
            if trust_remote_code: load_options["trust_remote_code"] = True
            if quantization_config: load_options["quantization_config"] = quantization_config
            # device_map auto usually needed for quantization or very large models
            # if quantization_config: load_options["device_map"] = "auto"
            if torch_dtype != torch.float32: load_options["torch_dtype"] = torch_dtype


            if model_type == "embedding":
                 model = SentenceTransformer(model_name, device=DEVICE, cache_folder=str(PARAMETERS["MODEL_CACHE_DIR"]))
                 # SentenceTransformer handles its own tokenization
                 tokenizer = None # No separate tokenizer needed
                 load_success = True
            elif model_type == "multimodal":
                 # Requires specific Processor and Model classes
                 if task == "image-to-text":
                     processor = BlipProcessor.from_pretrained(model_name, **load_options)
                     model = BlipForConditionalGeneration.from_pretrained(model_name, **load_options)
                     tokenizer = None # Processor includes tokenizer logic
                     load_success = True
                 else:
                     logger.error(f"Unsupported multimodal task '{task}' for model '{model_name}'")
            else: # Default text models
                 tokenizer = AutoTokenizer.from_pretrained(model_name, **load_options)
                 ModelClass = AutoModelForCausalLM if task == "text-generation" else \
                              AutoModelForQuestionAnswering if task == "question-answering" else \
                              AutoModel # Fallback
                 model = ModelClass.from_pretrained(model_name, **load_options)
                 load_success = True

            if load_success and model:
                 if hasattr(model, 'to'): # Check if model is a PyTorch module
                     model.to(DEVICE) # Move model to device
                     model.eval() # Set to evaluation mode
                 logger.info(f"✅ LLM '{model_key}' ({model_name}) loaded successfully to {DEVICE}. (Took {time.time() - load_start_time:.2f}s)")
            elif not load_success:
                 logger.error(f"Failed during loading process for '{model_key}'.")


        except Exception as e:
            logger.error(f"❌ Failed to load LLM '{model_key}' ({model_name}): {e}", exc_info=True)
            # Clean up partially loaded components
            del model, tokenizer, processor
            model, tokenizer, processor = None, None, None
            load_success = False


        # --- Create Pipeline (if applicable) ---
        pipeline_instance = None
        if load_success and model and task:
            try:
                # Pass components explicitly to pipeline constructor
                if model_type == "embedding":
                     pass # No standard pipeline for SentenceTransformer needed here
                elif processor:
                    pipeline_instance = pipeline(task, model=model, image_processor=processor, device=DEVICE)
                elif tokenizer:
                    pipeline_instance = pipeline(task, model=model, tokenizer=tokenizer, device=DEVICE)
                else: # Should not happen for standard HF models needing pipelines
                     logger.warning(f"Cannot create pipeline for {model_key}: Missing tokenizer/processor.")

                if pipeline_instance:
                     logger.debug(f"Pipeline created for {model_key} (Task: {task}).")
            except Exception as e:
                logger.error(f"Failed to create pipeline for {model_key} (Task: {task}): {e}", exc_info=False)


        # --- Update State ---
        if load_success:
            self.llm_models[model_key].update({
                "model": model,
                "tokenizer": tokenizer,
                "processor": processor,
                "pipeline": pipeline_instance,
                "last_used": time.time(),
            })
            self.llm_load_status[model_key] = True
            # Check memory again after successful load
            self._check_memory_usage(f"after loading {model_key}")
            return True
        else:
            # Ensure state reflects failure
            self.llm_models[model_key].update({"model": None, "tokenizer": None, "processor": None, "pipeline": None, "last_used": 0})
            self.llm_load_status[model_key] = False
            # Trigger garbage collection after failed load attempt
            gc.collect()
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
            return False


    def unload_llm_model(self, model_key: str, force: bool = False):
        """Unloads an LLM model to free memory."""
        if not self.is_llm_loaded(model_key): return

        # Prevent unloading essential embedding model unless forced?
        if model_key == "embedding" and not force:
             logger.debug("Skipping unload for essential embedding model.")
             return

        model_info = self.llm_models.get(model_key)
        if not model_info: return

        last_used = model_info.get("last_used", 0)
        unload_delay = PARAMETERS.get("MODEL_UNLOAD_DELAY", 900)

        if force or (time.time() - last_used > unload_delay):
            model_name = PARAMETERS.get(model_info["config_key"], "N/A")
            logger.info(f"Unloading LLM '{model_key}' ({model_name})...")
            try:
                # Explicitly delete model components
                if model_info["model"]: del model_info["model"]
                if model_info["tokenizer"]: del model_info["tokenizer"]
                if model_info["processor"]: del model_info["processor"]
                if model_info["pipeline"]: del model_info["pipeline"]

                # Reset the dictionary entry
                model_info.update({"model": None, "tokenizer": None, "processor": None, "pipeline": None, "last_used": 0})
                self.llm_load_status[model_key] = False

                # Force garbage collection and clear GPU cache
                gc.collect()
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()

                logger.info(f"✅ LLM '{model_key}' unloaded.")
                self._check_memory_usage(f"after unloading {model_key}")

            except Exception as e:
                logger.error(f"Error unloading LLM {model_key}: {e}", exc_info=True)
                # Ensure state is consistent even if error occurred during unload
                model_info.update({"model": None, "tokenizer": None, "processor": None, "pipeline": None, "last_used": 0})
                self.llm_load_status[model_key] = False

    def unload_inactive_llms(self):
        """Periodically checks and unloads inactive LLMs."""
        # logger.debug("Checking for inactive LLMs to unload...")
        # Iterate over a copy of keys as we might modify the dict
        for model_key in list(self.llm_models.keys()):
            self.unload_llm_model(model_key, force=False) # Use time threshold


    def get_llm_pipeline(self, model_key: str) -> Optional[Callable]:
        """Gets the pipeline for a loaded LLM, loading it if necessary."""
        if not self.load_llm_model(model_key): return None # Ensures model is loaded
        model_info = self.llm_models.get(model_key)
        if model_info:
             pipeline_instance = model_info.get("pipeline")
             if pipeline_instance:
                  model_info["last_used"] = time.time() # Update usage time
                  return pipeline_instance
             else:
                  logger.warning(f"Pipeline not available or not created for model '{model_key}'.")
                  return None
        return None

    def get_llm_model_and_tokenizer(self, model_key: str) -> Tuple[Optional[PreTrainedModel], Optional[Union[PreTrainedTokenizer, Any]]]:
        """Gets the model and tokenizer/processor, loading if necessary."""
        if not self.load_llm_model(model_key): return None, None
        model_info = self.llm_models.get(model_key)
        if model_info:
            model_info["last_used"] = time.time() # Update usage time
            model = model_info.get("model")
            # Return processor if it exists, otherwise tokenizer
            tokenizer_or_processor = model_info.get("processor") or model_info.get("tokenizer")
            # Special case for SentenceTransformer (model is the primary object)
            if model_key == "embedding" and isinstance(model, SentenceTransformer):
                 return model, None # Return model itself, no separate tokenizer needed
            return model, tokenizer_or_processor
        return None, None

    # --- Core Logic & Simulation Loop ---

    def run(self):
        """Starts the main execution logic."""
        if not self.initialized:
            logger.critical("NebulaAGI not initialized correctly. Cannot run.")
            return

        if PARAMETERS["UI_ENABLED"] and self.app and self.user_interface and self.nebula_thread:
             logger.info("Starting NebulaAGI background thread for continuous loop...")
             self.nebula_thread.start() # Start the loop in background thread
             logger.info("Starting UI event loop (blocking)...")
             # The UI thread will now handle interactions, calling methods on this NebulaAGI instance.
             sys.exit(self.app.exec()) # Start Qt event loop
        else:
             logger.info("Running NebulaAGI in headless mode...")
             self.continuous_learning_loop() # Run loop directly (blocking)

    @safe_loop(max_retries=5, delay=10) # Decorator for the main loop
    def continuous_learning_loop(self):
        """The main loop driving Nebula's operation, learning, and evolution."""
        if self.shutdown_requested: return # Check if shutdown requested before starting loop

        logger.info(f"🚀 Starting Continuous Learning Loop (Iteration {self.iteration})...")
        while not self.shutdown_requested:
            start_iter_time = time.time()
            self.iteration += 1
            logger.info(f"--- Iteration {self.iteration} ---")

            try:
                # --- Core Simulation Step ---
                if self.space:
                     self.space.run_simulation_step(self.iteration)

                # --- Information Acquisition & Processing ---
                if self.iteration % 5 == 0: # More frequent learning cycle
                    self.acquire_and_process_information()

                # --- Evolution ---
                if PARAMETERS["EVOLUTION_ENABLED"] and self.iteration % PARAMETERS["EVOLUTION_INTERVAL"] == 0:
                    self.evolve_system()

                # --- Self-Improvement/Correction Check ---
                if PARAMETERS["SELF_CORRECTION_ENABLED"] and self.iteration % PARAMETERS["SELF_IMPROVEMENT_INTERVAL"] == 0:
                    self.consider_self_improvement()

                # --- Monitoring & State ---
                if self.iteration % 20 == 0: # Update stats less frequently
                     self.display_statistics()
                if time.time() - self.last_backup_time > PARAMETERS["BACKUP_INTERVAL"]:
                    self.save_state()

                # --- Resource Management ---
                if self.iteration % 15 == 0: # Check memory periodically
                     self._check_memory_usage(f"Iteration {self.iteration} resource check")
                     self.unload_inactive_llms()

                # --- Iteration Timing ---
                iter_duration = time.time() - start_iter_time
                # Dynamic delay? Or fixed minimum step time?
                sleep_time = max(0.1, 1.0 - iter_duration) # Minimum 0.1s pause, aim for ~1s cycle
                # logger.debug(f"Iteration {self.iteration} took {iter_duration:.2f}s. Sleeping for {sleep_time:.2f}s.")
                time.sleep(sleep_time)

                # --- Emit stats for UI ---
                if PARAMETERS["UI_ENABLED"] and self.nebula_thread:
                     if self.iteration % 5 == 0: # Send stats every 5 iters
                         self.nebula_thread.stats_signal.emit(self.get_statistics_dict())


            except Exception as loop_e:
                 # This should ideally be caught by @safe_loop, but as a fallback:
                 logger.critical(f"💥 UNHANDLED EXCEPTION IN MAIN LOOP (Iteration {self.iteration}): {loop_e}", exc_info=True)
                 # Attempt to handle via error system
                 if self.error_correction_system:
                      try:
                          context = self.get_relevant_code_snippet(depth=1) # Get context from loop level
                      except: context = "Could not get loop context."
                      self.error_correction_system.handle_runtime_error(loop_e, context)
                 # Consider adding a longer pause after unhandled loop errors
                 time.sleep(30)

        logger.info("Continuous learning loop finished.")


    def acquire_and_process_information(self):
        """Acquires information (e.g., Wikipedia) and processes it."""
        if not PARAMETERS["ALLOW_INTERNET_ACCESS"]:
             logger.debug("Internet access disabled, skipping information acquisition.")
             return

        logger.info("Acquiring and processing information...")
        try:
            # Select a topic (maybe related to recent errors or low-knowledge areas?)
            topic = self._select_learning_topic()
            if not topic:
                 logger.warning("Could not select a topic for learning.")
                 return

            logger.info(f"Fetching info for topic: {topic}")
            info = self._retrieve_external_info(topic)

            if info:
                 logger.info(f"Processing information for '{topic}' (Length: {len(info)})...")
                 # Add to Knowledge Graph
                 self.learn_concepts_from_text(info)
                 # Add to Holographic Memory
                 self.holographic_memory.encode(topic, info) # Store full text under topic key
                 # Add summary to simple QA knowledge base?
                 # summary = self.generate_text(f"Summarize the following text:\n{info}", model_key="text_generation_small", max_length=100)
                 # if summary: self.add_to_knowledge_base(f"What is {topic}?", summary)

        except Exception as e:
            logger.error(f"Error during information acquisition/processing: {e}", exc_info=False)
            self.handle_error(str(e), self.get_relevant_code_snippet())

    def _select_learning_topic(self) -> Optional[str]:
        """Selects a topic for information acquisition."""
        # Strategy: Prioritize topics related to recent errors, then KG gaps, then random.
        # 1. Error-related topics (if error history exists)
        if self.error_history:
             last_error = self.error_history[-1]
             # Extract keywords from error message (simple approach)
             keywords = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', last_error.get('message', ''))
             potential_topics = [k for k in keywords if len(k) > 4 and k not in ['self', 'error', 'message', 'traceback', 'line']]
             if potential_topics:
                  topic = random.choice(potential_topics)
                  logger.debug(f"Selecting topic based on recent error: {topic}")
                  return topic

        # 2. Explore sparse areas of Knowledge Graph (nodes with few connections)
        if self.knowledge_graph and self.knowledge_graph.graph.number_of_nodes() > 10:
            degrees = dict(self.knowledge_graph.graph.degree())
            if degrees:
                min_degree_node = min(degrees, key=degrees.get)
                # Check if degree is low (e.g., < 2)
                if degrees[min_degree_node] < 2:
                    logger.debug(f"Selecting topic from sparse KG area: {min_degree_node}")
                    return min_degree_node

        # 3. Random topic from a predefined list
        default_topics = [
            "Quantum Computing", "Optical Neural Networks", "Holography", "AGI Safety",
            "Neuroscience Memory Formation", "Evolutionary Algorithms", "Self-modifying Systems",
            "Large Language Models", "Python Programming Best Practices", "PyTorch Optimization",
            "Complex Systems", "Emergence", "Information Theory"
        ]
        topic = random.choice(default_topics)
        logger.debug(f"Selecting random default topic: {topic}")
        return topic

    def _retrieve_external_info(self, topic: str) -> Optional[str]:
        """Retrieves information from external sources like Wikipedia."""
        # Currently only uses Wikipedia
        try:
            logger.debug(f"Querying Wikipedia for '{topic}'...")
            wikipedia.set_lang(PARAMETERS.get("WIKIPEDIA_LANG", "en"))
            # Get a few paragraphs, auto_suggest helps find related pages
            summary = wikipedia.summary(topic, sentences=10, auto_suggest=True, redirect=True)
            return summary
        except wikipedia.exceptions.PageError:
            logger.warning(f"Wikipedia page not found for: {topic}")
            return None
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(f"Wikipedia disambiguation for '{topic}'. Choosing first option: {e.options[0]}")
            try: # Try fetching the first disambiguation option
                return wikipedia.summary(e.options[0], sentences=10, auto_suggest=False, redirect=True)
            except Exception:
                 logger.error(f"Failed to fetch disambiguated page '{e.options[0]}'.")
                 return None
        except requests.exceptions.ConnectionError:
             logger.error("Wikipedia connection error. Check internet connection.")
             PARAMETERS["ALLOW_INTERNET_ACCESS"] = False # Disable future attempts
             return None
        except Exception as e:
            logger.error(f"Error fetching Wikipedia info for '{topic}': {e}", exc_info=False)
            return None

    @require_component('knowledge_graph')
    @require_component('spacy_nlp')
    def learn_concepts_from_text(self, text: str):
        """Extracts concepts/relations from text and adds to Knowledge Graph."""
        if not text: return
        logger.info(f"Learning concepts from text (length: {len(text)})...")
        try:
            # Process only first N characters for performance
            doc = self.spacy_nlp(text[:15000])
            concepts = set()
            # Extract Noun Chunks and Entities as potential concepts
            concepts.update(chunk.text.lower() for chunk in doc.noun_chunks if 3 < len(chunk.text) < 40)
            entities = {ent.text.lower() for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "NORP", "FAC"}}
            concepts.update(entities)

            added_concepts = 0
            for concept in concepts:
                 # Basic filtering
                 if re.fullmatch(r'[a-z0-9\s\-]+', concept) and len(concept.split()) <= 5:
                     self.knowledge_graph.add_concept(concept)
                     added_concepts +=1

            # Extract simple relations (Subject-Verb-Object triples) - Very basic
            relations_added = 0
            for sent in doc.sents:
                 subj = None
                 verb = None
                 obj = None
                 # Simplified extraction - look for nsubj, ROOT verb, dobj/pobj
                 for token in sent:
                      if "subj" in token.dep_ and subj is None: # Take first subject
                           subj = token.lemma_.lower()
                      elif "ROOT" in token.dep_ and token.pos_ == "VERB" and verb is None:
                           verb = token.lemma_.lower()
                      elif ("obj" in token.dep_ or "attr" in token.dep_) and obj is None: # Direct object or attribute
                           obj = token.lemma_.lower()

                      # If we have a basic triple, add it if nodes exist in our concepts
                      if subj and verb and obj:
                           # Check if extracted subj/obj roughly match known concepts
                           # This is weak, better relation extraction needed for accuracy
                           if subj in concepts and obj in concepts:
                                self.knowledge_graph.add_relation(subj, obj, verb)
                                relations_added += 1
                           # Reset for next potential triple in the sentence
                           subj, verb, obj = None, None, None # Reset


            logger.info(f"KG Update: Added/updated {added_concepts} concepts and {relations_added} potential relations.")

        except Exception as e:
            logger.error(f"Error learning concepts from text: {e}", exc_info=False)


    # --- Core Interaction Methods ---

    @require_component('holographic_memory')
    @require_component('knowledge_graph')
    def answer_question(self, question: str) -> str:
        """Answers a question using KG, Holographic Memory, or LLM."""
        if not question: return "Please ask a question."
        logger.info(f"Answering question: '{question}'")
        answer = "I'm sorry, I cannot answer that question at the moment." # Default

        try:
            # 1. Try Holographic Memory (Associative Recall)
            holo_results = self.holographic_memory.decode(question, top_k=1)
            if holo_results:
                 key, data, score = holo_results[0]
                 logger.info(f"Found relevant info in Holographic Memory (Key: '{key}', Score: {score:.3f})")
                 # Format the retrieved data as an answer
                 answer = f"Recalling information related to '{key}':\n{str(data)[:1000]}..." # Limit answer length
                 return answer

            # 2. Try Knowledge Graph Reasoning/Search
            kg_results = self.knowledge_graph.reason(question) # Or find_similar_concepts
            # Check if KG returned something more than the default failure message
            if kg_results and "not found" not in kg_results[0].lower() and "could not derive" not in kg_results[0].lower() :
                 logger.info("Found potentially relevant info in Knowledge Graph.")
                 answer = "From my knowledge graph:\n" + "\n".join(kg_results)[:1000]
                 return answer

            # 3. Fallback to QA LLM (if available)
            logger.info("No direct match in memory/KG. Querying QA LLM...")
            qa_pipeline = self.get_llm_pipeline("qa")
            if qa_pipeline:
                 # Provide context if possible (e.g., from KG reasoning results)
                 context = (kg_results[0] if kg_results and len(kg_results[0]) > 20 else # Use KG result if substantial
                           "General knowledge context.")[:1500] # Limit context length
                 try:
                      qa_result = qa_pipeline(question=question, context=context)
                      if qa_result and qa_result.get('answer') and qa_result.get('score', 0) > 0.1: # Basic confidence check
                           logger.info(f"QA LLM provided answer (Score: {qa_result['score']:.3f})")
                           answer = qa_result['answer']
                           # Optional: Add good QA results back to memory/KG?
                           # self.add_to_knowledge_base(question, answer)
                           return answer
                      else:
                           logger.warning("QA LLM returned low confidence or no answer.")
                 except Exception as e:
                      logger.error(f"Error querying QA pipeline: {e}", exc_info=False)
                      # Fall through to generative model if QA fails

            # 4. Final Fallback: Generative LLM
            logger.info("QA failed or unavailable. Using generative LLM as final fallback.")
            answer = self.generate_text(f"Please answer the following question based on general knowledge: {question}",
                                        model_key="text_generation_large") # Use large model for better answers
            return answer

        except Exception as e:
             logger.error(f"Unexpected error during question answering: {e}", exc_info=True)
             return "I encountered an internal error trying to answer your question."


    @require_llm('text_generation_small') # Decorator ensures small model is loaded
    def generate_text(self, prompt: str, model_key: str = "text_generation_small", max_length: Optional[int] = None) -> str:
        """Generates text using the specified LLM."""
        if not prompt: return ""
        # Determine max_length based on parameter or default
        eff_max_length = max_length if max_length is not None else PARAMETERS["MAX_LLM_GENERATION_LENGTH"]

        # Load the requested model (decorator handles this, but double check key)
        if model_key not in self.llm_models:
             logger.error(f"Invalid model key '{model_key}' for text generation.")
             return f"[Error: Invalid model key '{model_key}']"
        # Ensure the specific model needed is loaded
        if not self.load_llm_model(model_key):
             logger.error(f"Could not load model '{model_key}' for text generation.")
             return f"[Error: Model '{model_key}' unavailable]"

        generator_pipeline = self.get_llm_pipeline(model_key)
        if not generator_pipeline:
             # Try using model directly if pipeline failed but model loaded
             model, tokenizer = self.get_llm_model_and_tokenizer(model_key)
             if model and tokenizer:
                  logger.warning(f"Pipeline for '{model_key}' failed. Attempting direct generation.")
                  try:
                       inputs = tokenizer(prompt, return_tensors="pt", max_length=PARAMETERS["MAX_LLM_INPUT_LENGTH"], truncation=True).to(DEVICE)
                       # Calculate max_new_tokens based on desired total length and input length
                       max_new_tokens = eff_max_length - inputs.input_ids.shape[1]
                       if max_new_tokens <= 0: return "[Input already too long]"

                       output_sequences = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            num_return_sequences=1,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id # Important for stopping generation
                       )
                       generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
                       # Remove prompt from beginning
                       if generated_text.startswith(prompt):
                           generated_text = generated_text[len(prompt):].strip()
                       return generated_text
                  except Exception as e:
                       logger.error(f"Error during direct text generation with {model_key}: {e}", exc_info=False)
                       return f"[Error generating text with {model_key}]"
             else:
                  logger.error(f"Pipeline and direct model/tokenizer unavailable for '{model_key}'.")
                  return f"[Error: Generation model '{model_key}' completely unavailable]"

        # Use pipeline if available
        logger.info(f"Generating text (model: {model_key}, max_len: {eff_max_length}) for prompt: '{prompt[:70]}...'")
        try:
            # Calculate max_new_tokens based on prompt length if pipeline needs it
            # Note: Many pipelines handle total length directly via max_length
            # We might need to tokenize the prompt to find its length accurately
            # For simplicity, we assume pipeline's max_length is total length
            results = generator_pipeline(
                 prompt,
                 max_length=eff_max_length, # Total length including prompt
                 num_return_sequences=1,
                 do_sample=True,
                 temperature=0.7,
                 top_p=0.9,
                 truncation=True, # Ensure input prompt is truncated if needed
                 pad_token_id=generator_pipeline.tokenizer.eos_token_id if hasattr(generator_pipeline, 'tokenizer') else 50256 # Default EOS
                 )

            if results and isinstance(results, list):
                generated_text = results[0]['generated_text']
                # Remove prompt from beginning (pipelines often include it)
                if generated_text.startswith(prompt):
                     generated_text = generated_text[len(prompt):].strip()
                logger.debug(f"Generated text: '{generated_text[:100]}...'")
                self.llm_interaction_log.append({"prompt": prompt, "response": generated_text})
                return generated_text
            else:
                 logger.warning(f"Text generation pipeline for {model_key} returned unexpected format: {results}")
                 return "[Error: Unexpected generation result]"

        except Exception as e:
            logger.error(f"Error generating text with pipeline {model_key}: {e}", exc_info=False)
            self.handle_error(str(e), f"Error in generate_text pipeline {model_key}")
            return f"[Error generating text with {model_key}]"


    # --- Evolution & Self-Modification ---

    def _setup_genetic_algorithm(self) -> Optional[base.Toolbox]:
        """Sets up the DEAP genetic algorithm toolbox."""
        if not DEAP_AVAILABLE: return None
        logger.debug("Setting up Genetic Algorithm Toolbox...")
        try:
            # Ensure FitnessMin/Individual are defined only once or handle redefinition
            if hasattr(creator, "FitnessMin"): del creator.FitnessMin
            if hasattr(creator, "Individual"): del creator.Individual

            creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Minimize negative performance
            creator.create("Individual", list, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()
            param_keys = list(NebulaGenome.parameter_ranges.keys())

            # Gene generator function: creates one gene (parameter value)
            def create_gene(param_index):
                 param_name = param_keys[param_index]
                 min_val, max_val = NebulaGenome.parameter_ranges[param_name]
                 if param_name in NebulaGenome.int_params:
                      return random.randint(min_val, max_val)
                 else:
                      return random.uniform(min_val, max_val)

            # Register attributes (genes) for the individual
            num_genes = len(param_keys)
            toolbox.register("attr_gene", create_gene) # Needs index argument now
            # Use initCycle with proper arguments for DEAP >= 1.4
            toolbox.register("individual", tools.initCycle, creator.Individual,
                         (lambda: toolbox.attr_gene(i) for i in range(num_genes)), n=1)

            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            # Genetic operators
            toolbox.register("evaluate", self.evaluate_genome_fitness)
            toolbox.register("mate", tools.cxBlend, alpha=0.5) # Blend crossover for floats
            toolbox.register("mutate", self.mutate_genome_individual, indpb=0.2) # Use custom mutation
            toolbox.register("select", tools.selTournament, tournsize=3)

            logger.debug("Genetic Algorithm Toolbox setup complete.")
            return toolbox
        except Exception as e:
            logger.error(f"Failed to setup DEAP toolbox: {e}", exc_info=True)
            return None

    def mutate_genome_individual(self, individual: creator.Individual, indpb: float) -> Tuple[creator.Individual]:
        """Custom mutation function for the mixed-type genome list."""
        param_keys = list(NebulaGenome.parameter_ranges.keys())
        for i in range(len(individual)):
             if random.random() < indpb:
                 param_name = param_keys[i]
                 min_val, max_val = NebulaGenome.parameter_ranges[param_name]
                 if param_name in NebulaGenome.int_params:
                      # Mutate integer: small random change
                      change = random.randint(-max(1, int((max_val - min_val) * 0.1)), max(1, int((max_val-min_val)*0.1))) # Mutate by up to 10% of range
                      individual[i] = np.clip(individual[i] + change, min_val, max_val)
                 else:
                      # Mutate float: Gaussian perturbation
                      sigma = (max_val - min_val) * 0.1 # Std dev is 10% of range
                      individual[i] += random.gauss(0, sigma)
                      individual[i] = np.clip(individual[i], min_val, max_val)
        return individual, # Return as tuple


    def evaluate_genome_fitness(self, individual_list: List) -> Tuple[float]:
        """Evaluates the fitness of a genome by running simulation tasks."""
        start_eval_time = time.time()
        # Convert list to Genome object
        temp_genome = NebulaGenome.from_list(individual_list)

        # --- Evaluation Strategy: Temporarily apply, run tasks, restore ---
        # Backup original parameters controlled by the genome
        original_params = {}
        for param_name in NebulaGenome.parameter_ranges.keys():
             param_key = param_name.upper()
             if param_key in PARAMETERS:
                  original_params[param_key] = PARAMETERS[param_key]
        # Store original neuron count
        original_neuron_count = len(self.space.neurons) if self.space else 0

        try:
            # Apply temporary genome to global PARAMETERS
            temp_genome.apply_to_parameters()
            # Adjust neuron population if needed (this part is tricky for temporary eval)
            # For simplicity, we might skip population adjustment during fitness eval,
            # or accept the side effect. Let's apply it for now.
            # self.adjust_population_size(int(temp_genome.population_size)) # This might affect main state too much

            # Run evaluation tasks (use fewer tasks for speed)
            performance_score = self._run_evaluation_tasks(num_tasks=PARAMETERS["FITNESS_EVAL_TASKS"])

            # Fitness: Higher performance is better. Minimize (1.0 - performance).
            # Add penalty for instability? (e.g., if errors occurred during eval tasks)
            # Add penalty for extreme resource usage during eval?
            fitness_value = 1.0 - performance_score

        except Exception as e:
            logger.error(f"Error during fitness evaluation: {e}", exc_info=False)
            fitness_value = 2.0 # Penalize heavily (worse than max negative performance)
        finally:
            # --- Restore original parameters ---
            logger.debug("Restoring original parameters after fitness evaluation...")
            for param_key, value in original_params.items():
                PARAMETERS[param_key] = value
            # Restore neuron population? This is harder. Maybe just log the change?
            # self.adjust_population_size(original_neuron_count) # This could be disruptive

        eval_duration = time.time() - start_eval_time
        logger.debug(f"Evaluated genome fitness: {1.0 - fitness_value:.4f} (Duration: {eval_duration:.2f}s)")
        return (fitness_value,) # Return fitness as a tuple

    def _run_evaluation_tasks(self, num_tasks: int = 5) -> float:
        """Runs predefined tasks to get a performance score (0-1)."""
        scores = []
        logger.debug(f"Running {num_tasks} evaluation tasks...")
        possible_tasks = ["qa", "generate", "retrieve", "kg_query"] # Add more? "image_caption"?

        for i in range(num_tasks):
            task_type = random.choice(possible_tasks)
            score = 0.0
            task_start_time = time.time()
            try:
                if task_type == "qa":
                     # Ask a question based on known concepts or random
                     if self.knowledge_graph and self.knowledge_graph.graph.number_of_nodes() > 0:
                          concepts = list(self.knowledge_graph.graph.nodes())
                          q = f"What can you tell me about {random.choice(concepts)}?"
                     else: q = "What is artificial intelligence?"
                     answer = self.answer_question(q)
                     # Simple scoring: non-empty, not default error, reasonable length
                     if answer and "cannot answer" not in answer.lower() and "internal error" not in answer.lower() and len(answer) > 20:
                          score = 0.8
                     else: score = 0.1
                elif task_type == "generate":
                     prompt = "Write a short paragraph about the future of AI."
                     text = self.generate_text(prompt, model_key="text_generation_small", max_length=80)
                     # Simple scoring: check length and basic coherence (presence of nouns/verbs?)
                     if text and len(text) > 30:
                          # Use SpaCy for basic check if available
                          if self.spacy_nlp:
                               doc = self.spacy_nlp(text)
                               if any(tok.pos_ in ("NOUN", "PROPN") for tok in doc) and any(tok.pos_ == "VERB" for tok in doc):
                                    score = 0.7
                               else: score = 0.3
                          else: score = 0.5 # Less strict if no SpaCy
                     else: score = 0.1
                elif task_type == "retrieve":
                     # Encode something, then try retrieve
                     test_key = f"eval_key_{random.randint(1000,9999)}"
                     test_data = f"Evaluation data {time.time()}"
                     if self.holographic_memory:
                          self.holographic_memory.encode(test_key, test_data)
                          results = self.holographic_memory.decode(test_key, threshold=0.8) # High threshold for exact match
                          if results and results[0][1] == test_data:
                               score = 0.9
                          else: score = 0.1
                     else: score = 0.0 # Cannot perform task
                elif task_type == "kg_query":
                     if self.knowledge_graph and self.knowledge_graph.graph.number_of_nodes() > 0:
                          concepts = list(self.knowledge_graph.graph.nodes())
                          concept = random.choice(concepts)
                          related = self.knowledge_graph.get_related_concepts(concept, max_distance=1)
                          # Score based on finding any related concepts
                          score = 0.7 if related else 0.2
                     else: score = 0.1 # KG empty or unavailable

                task_duration = time.time() - task_start_time
                logger.debug(f" Eval task '{task_type}' -> Score: {score:.2f} ({task_duration:.2f}s)")

            except Exception as e:
                logger.warning(f"Error during evaluation task '{task_type}': {e}", exc_info=False)
                score = 0.0 # Penalize if task itself fails
            scores.append(score)

        avg_score = np.mean(scores) if scores else 0.0
        logger.debug(f"Finished evaluation tasks. Average Performance Score: {avg_score:.4f}")
        # Record performance
        self.performance_history.append(avg_score)
        return avg_score

    # Apply Genome is handled by NebulaGenome.apply_to_parameters() now

    @require_component('space')
    def adjust_population_size(self, new_size: int):
        """Adjusts the number of neurons in NebulaSpace."""
        # This should be called cautiously, potentially not during fitness eval
        current_size = len(self.space.neurons)
        min_neurons = PARAMETERS.get("MIN_NEURONS", 50)
        max_neurons = PARAMETERS["MAX_NEURONS"]
        target_size = int(np.clip(new_size, min_neurons, max_neurons))
        delta = target_size - current_size

        if delta == 0: return
        logger.info(f"Adjusting neuron population from {current_size} to {target_size} (Delta: {delta})...")

        if delta > 0: # Add neurons
            for _ in range(delta):
                neuron_id = self.space._get_new_id("neuron")
                position = torch.rand(3, device=self.device) * self.space.dimensions
                try:
                     neuron = QuantumNeuron(
                          neuron_id=neuron_id, position=position, device=self.device # Use defaults
                     )
                     self.space.add_neuron(neuron) # Handles structure assignment
                except Exception as e:
                     logger.error(f"Failed to add neuron {neuron_id} during population adjustment: {e}")
                     self.space.next_neuron_id -= 1 # Roll back ID

        elif delta < 0: # Remove neurons
             num_to_remove = abs(delta)
             # Strategy: remove lowest luminosity neurons first
             if self.space.neurons:
                 sorted_neurons = sorted(self.space.neurons.values(), key=lambda n: n.luminosity.item())
                 ids_to_remove = [n.id for n in sorted_neurons[:num_to_remove]]
                 logger.info(f"Removing {len(ids_to_remove)} neurons (lowest luminosity).")
                 for nid in ids_to_remove:
                      self.space.remove_neuron(nid)

        self.space.kdtree_needs_update = True
        logger.info(f"Neuron population adjusted to {len(self.space.neurons)}.")

    @require_component('genetic_algorithm')
    def evolve_system(self):
        """Runs one cycle of the genetic algorithm."""
        if not PARAMETERS["EVOLUTION_ENABLED"]: return
        current_time = time.time()
        # Allow evolution always if interval is 0, otherwise check time
        if PARAMETERS["EVOLUTION_INTERVAL"] > 0 and current_time - self.last_evolution_time < 300: # Min 5 mins between forced cycles
              # Check if interval (in iterations) has passed
              if self.iteration % PARAMETERS["EVOLUTION_INTERVAL"] != 0:
                  return # Skip if interval (iterations) not reached

        logger.info("🧬 Starting Evolution Cycle...")
        self.last_evolution_time = current_time
        try:
            toolbox = self.genetic_algorithm
            pop_size = PARAMETERS["GA_POPULATION_SIZE"]
            num_gen = PARAMETERS["GA_GENERATIONS"]
            cxpb = PARAMETERS["GA_CXPB"]
            mutpb = PARAMETERS["GA_MUTPB"]

            # Initialize population
            population = toolbox.population(n=pop_size)

            # Evaluate the initial population
            fitnesses = map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Statistics (optional)
            stats = tools.Statistics(lambda ind: ind.fitness.values[0])
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)

            # Run the algorithm
            logger.info(f"Running GA: Population={pop_size}, Generations={num_gen}")
            result_pop, logbook = algorithms.eaSimple(population, toolbox, cxpb, mutpb, num_gen,
                                                      stats=stats, verbose=False) # verbose=True for generation stats

            # Select the best individual from the final population
            best_individual = tools.selBest(result_pop, k=1)[0]
            best_fitness = best_individual.fitness.values[0]

            logger.info(f"Evolution finished. Best Fitness (1-Performance): {best_fitness:.4f}")
            # Convert best individual list back to Genome object
            new_genome = NebulaGenome.from_list(best_individual)

            # Apply the best genome found
            logger.info("Applying best evolved genome...")
            self.genome = new_genome # Update the main genome
            self.genome.apply_to_parameters() # Apply to global PARAMETERS

        except Exception as e:
            logger.error(f"Error during evolution cycle: {e}", exc_info=True)
            self.handle_error(f"Evolution cycle failed: {e}", self.get_relevant_code_snippet())

    def consider_self_improvement(self):
        """Considers analyzing and potentially modifying its own code."""
        if not PARAMETERS["SELF_CORRECTION_ENABLED"] or not self.error_correction_system:
            return
        current_time = time.time()
        if current_time - self.last_improvement_check_time < 600: # Check every 10 mins max
             if self.iteration % PARAMETERS["SELF_IMPROVEMENT_INTERVAL"] != 0:
                 return

        logger.info("🤔 Considering self-improvement...")
        self.last_improvement_check_time = current_time

        try:
            # Strategy: Analyze a random part of the code? Or focus on areas with past errors?
            # Simple: Analyze overall code quality (placeholder)
            full_code = self.get_full_code_text()
            if not full_code: return

            quality_score = self.error_correction_system.learner.analyze_code_quality(full_code)
            logger.info(f"Code quality analysis score (placeholder): {quality_score:.3f}")

            # If quality is low or errors exist, try to suggest improvements
            # This part is highly conceptual and risky
            if quality_score < 0.6 or len(self.error_history) > 0:
                logger.info("Attempting to find area for improvement...")
                # Select a code snippet to try and improve (e.g., a random function)
                snippet_context = self._get_random_function_code()
                if not snippet_context: return

                # Use LLM to suggest improvement (similar to correction but different prompt)
                improvement_prompt = f"""Analyze the following Python code snippet from the NebulaAGI system:
--- CODE SNIPPET ---
{snippet_context}

--- TASK ---
Suggest potential improvements for clarity, efficiency, or robustness. Output ONLY the improved Python code snippet. If no improvements seem necessary, output the original snippet. Do not include explanations.
Improved Code Snippet:
"""
                suggestion = self._query_codegen_llm(improvement_prompt)

                if suggestion and suggestion != snippet_context.strip(): # Only proceed if suggestion is different
                     logger.info(f"Suggested improvement:\n{suggestion}")
                     # Test the suggested improvement
                     test_passed, test_output = self.error_correction_system.tester.test_snippet(suggestion)
                     logger.info(f"Improvement test result: Passed={test_passed}")
                     if test_passed:
                          # Apply the improvement (RISKY!)
                          apply_success = self.apply_code_modification(snippet_context, suggestion)
                          if apply_success:
                               logger.info("✅ Improvement applied successfully.")
                               self.modification_history.append({
                                    "type": "improvement", "timestamp": time.time(),
                                    "original": snippet_context, "modified": suggestion
                               })
                          else:
                               logger.error("Failed to apply tested improvement.")

        except Exception as e:
             logger.error(f"Error during self-improvement consideration: {e}", exc_info=False)
             self.handle_error(f"Self-improvement failed: {e}", self.get_relevant_code_snippet())


    def _get_random_function_code(self) -> Optional[str]:
        """Extracts the source code of a random method within NebulaAGI class."""
        try:
             methods = inspect.getmembers(self, predicate=inspect.ismethod)
             # Filter out private methods or methods not defined in this class
             nebula_methods = [m for name, m in methods if inspect.getmodule(m) == inspect.getmodule(self.__class__) and not name.startswith('_')]
             if not nebula_methods: return None
             random_method = random.choice(nebula_methods)
             source_code = inspect.getsource(random_method)
             return source_code
        except Exception as e:
             logger.error(f"Failed to get source code for random method: {e}")
             return None

    def _query_codegen_llm(self, prompt: str) -> Optional[str]:
        """Helper to query the code generation LLM."""
        if not self.error_correction_system or not self.error_correction_system.corrector or not self.error_correction_system.corrector.codegen_pipeline:
            logger.error("CodeGen LLM pipeline not available for querying.")
            return None
        pipeline = self.error_correction_system.corrector.codegen_pipeline
        try:
            # Adjust max_length dynamically based on prompt
            # Estimate tokens roughly (e.g., 3 chars per token)
            prompt_tokens = len(prompt) // 3
            max_len = prompt_tokens + 300 # Allow 300 new tokens for suggestion
            results = pipeline(prompt, max_length=max_len, num_return_sequences=1, temperature=0.2, truncation=True)
            if results and isinstance(results, list):
                 generated_text = results[0]['generated_text']
                 # Extract part after prompt (specific to prompt structure)
                 suggestion = re.split(r"Improved Code Snippet:|Corrected Code Snippet:", generated_text)[-1]
                 suggestion = re.sub(r"```python\n?|```", "", suggestion).strip()
                 if suggestion: return suggestion
            return None
        except Exception as e:
             logger.error(f"Error querying CodeGen LLM: {e}", exc_info=False)
             return None


    def get_full_code_text(self) -> Optional[str]:
        """Reads the source code of the current file."""
        try:
            with open(__file__, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read own source code: {e}")
            return None

    def get_relevant_code_snippet(self, depth=1) -> str:
        """Tries to get code snippet from the call stack."""
        try:
            frame = inspect.currentframe()
            for _ in range(depth + 1): # Go up the stack
                if frame.f_back:
                    frame = frame.f_back
                else: break # Stop if no more frames up

            filename = inspect.getfile(frame)
            lineno = frame.f_lineno
            if filename != __file__: return f"Error occurred outside main script ({filename})"

            full_code = self.get_full_code_text()
            if full_code:
                lines = full_code.splitlines()
                start = max(0, lineno - 8)
                end = min(len(lines), lineno + 7)
                snippet = f"... (Lines {start+1}-{end} of {Path(__file__).name}) ...\n"
                snippet += "\n".join(f"{i+1:4d}{'>' if i+1==lineno else ' '} {line}" for i, line in enumerate(lines[start:end]))
                snippet += "\n..."
                return snippet
            else: return "Could not read source code."
        except Exception as e:
            return f"Could not get code snippet: {e}"
        finally:
             del frame # Avoid reference cycles


    def handle_error(self, error_msg: str, code_context: Optional[str] = None, exception: Optional[Exception] = None):
        """Logs an error and potentially triggers self-correction."""
        logger.error(f"Error handled: {error_msg}")
        error_data = {
            "timestamp": time.time(),
            "message": error_msg,
            "context": code_context or "No context provided.",
            "exception_type": type(exception).__name__ if exception else "N/A"
        }
        self.error_history.append(error_data)

        # Trigger self-correction if enabled and system available
        if PARAMETERS["SELF_CORRECTION_ENABLED"] and self.error_correction_system and exception:
            logger.info("Forwarding error to self-correction system...")
            # Run correction attempt in a separate thread? Or directly?
            # Running directly might block the main loop.
            # For now, run directly.
            try:
                 self.error_correction_system.handle_runtime_error(exception, code_context)
            except Exception as correction_e:
                 logger.error(f"Error occurred *within* the error handling system itself: {correction_e}", exc_info=True)

    def apply_code_modification(self, original_snippet: str, new_snippet: str) -> bool:
        """
        Replaces the original_snippet with new_snippet in the source file.
        WARNING: EXTREMELY DANGEROUS. Modifies the running script file.
        """
        logger.warning("!!! ATTEMPTING DANGEROUS SELF-MODIFICATION !!!")
        original_snippet = original_snippet.strip()
        new_snippet = new_snippet.strip()
        if not original_snippet or not new_snippet:
             logger.error("Cannot apply modification: original or new snippet is empty.")
             return False
        if original_snippet == new_snippet:
             logger.info("No changes detected in modification snippet. Skipping application.")
             return False

        try:
            file_path = Path(__file__)
            with file_path.open('r', encoding='utf-8') as f:
                full_code = f.read()

            # Normalize line endings for comparison
            original_snippet_norm = "\n".join(original_snippet.splitlines())
            full_code_norm = "\n".join(full_code.splitlines())

            # Find the original snippet (handle potential indentation issues naively)
            start_index = full_code_norm.find(original_snippet_norm)

            if start_index == -1:
                # Try finding with flexible whitespace matching (more complex)
                # Simple fallback: maybe only first line matches due to LLM formatting?
                first_line_orig = original_snippet_norm.splitlines()[0].strip()
                start_index = full_code_norm.find(first_line_orig)
                if start_index != -1:
                     # If found via first line, assume replacement length is approx original snippet line count
                     num_lines_orig = len(original_snippet_norm.splitlines())
                     # Find the actual end index in the full code
                     end_index = start_index
                     lines_found = 0
                     while lines_found < num_lines_orig and end_index < len(full_code_norm):
                          if full_code_norm[end_index] == '\n':
                               lines_found += 1
                          end_index += 1
                     logger.warning("Found original snippet using fuzzy match (first line). Replacement might be imprecise.")

                else:
                    logger.error("Could not find the exact original code snippet in the source file. Modification aborted.")
                    # Log snippets for debugging
                    logger.debug(f"--- Original Snippet (Normalized) ---\n{original_snippet_norm}\n------------------------------------")
                    return False

            else:
                 end_index = start_index + len(original_snippet_norm)


            # Construct the new full code
            new_full_code = full_code_norm[:start_index] + new_snippet + full_code_norm[end_index:]

            # Write the modified code back to the file
            # Make a backup first!
            backup_path = file_path.with_suffix(".py.bak")
            logger.info(f"Backing up current code to {backup_path}")
            file_path.rename(backup_path)

            logger.warning(f"Writing modified code to {file_path}...")
            with file_path.open('w', encoding='utf-8') as f:
                # Write with OS-specific line endings? Or stick to \n? Stick to \n.
                f.write(new_full_code)

            logger.info("Code modification written. Relaunching Nebula is likely required for changes to take full effect.")
            # Ideally, the system should restart itself here, but that's complex.
            # For now, log modification and continue (changes might partially apply via importlib later?)

            self.modification_history.append({
                "timestamp": time.time(), "type": "correction/improvement",
                "original_hash": hashlib.md5(original_snippet.encode()).hexdigest(),
                "new_hash": hashlib.md5(new_snippet.encode()).hexdigest()
                })
            return True

        except Exception as e:
            logger.critical(f"CRITICAL ERROR during code modification: {e}", exc_info=True)
            # Attempt to restore backup?
            if backup_path.exists():
                 logger.warning(f"Attempting to restore backup from {backup_path}...")
                 try:
                     backup_path.rename(file_path)
                     logger.info("Backup restored.")
                 except Exception as restore_e:
                      logger.critical(f"FAILED TO RESTORE BACKUP: {restore_e}")
            return False


    # --- State Management ---
    def save_state(self):
        """Saves the complete state of the NebulaAGI system."""
        logger.info(f"Saving Nebula state to {PARAMETERS['STATE_FILE']}...")
        state = {
            "version": self.VERSION,
            "timestamp": time.time(),
            "iteration": self.iteration,
            "space_state": self.space.get_space_state() if self.space else None,
            "knowledge_graph_state": self.knowledge_graph.get_state() if self.knowledge_graph else None,
            "holographic_memory_state": self.holographic_memory.get_state() if self.holographic_memory else None,
            "genome_state": self.genome.to_list() if self.genome else None, # Save genome as list
            "error_history": list(self.error_history),
            "modification_history": list(self.modification_history),
            "performance_history": list(self.performance_history),
            # Save next IDs from space directly
            "next_neuron_id": self.space.next_neuron_id if self.space else 0,
            "next_cluster_id": self.space.next_cluster_id if self.space else 0,
            "next_sector_id": self.space.next_sector_id if self.space else 0,
        }
        try:
            # Use pickle for saving complex objects like NetworkX graph, modules etc.
            # Add backup mechanism
            state_file = PARAMETERS["STATE_FILE"]
            backup_file = PARAMETERS["BACKUP_DIRECTORY"] / f"nebula_state_backup_{time.strftime('%Y%m%d_%H%M%S')}.pkl"

            # Save to temp file first
            temp_file = state_file.with_suffix(".pkl.tmp")
            with open(temp_file, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Create backup of existing state file if it exists
            if state_file.exists():
                 try:
                    shutil.copyfile(state_file, backup_file)
                    logger.info(f"Created backup of previous state: {backup_file}")
                 except Exception as bk_e:
                      logger.warning(f"Could not create state backup: {bk_e}")


            # Atomically replace old state file with new one
            os.replace(temp_file, state_file) # Use os.replace for atomicity (on most OS)

            self.last_backup_time = time.time()
            logger.info("Nebula state saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save Nebula state: {e}", exc_info=True)
            # Clean up temp file if save failed
            if temp_file.exists(): os.remove(temp_file)

    def load_state(self):
        """Loads the system state from the file."""
        state_file = PARAMETERS["STATE_FILE"]
        if state_file.exists():
            logger.info(f"Loading Nebula state from {state_file}...")
            try:
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)

                # Check version compatibility?
                loaded_version = state.get("version", "N/A")
                if loaded_version != self.VERSION:
                    logger.warning(f"Loading state from different version (State: {loaded_version}, Current: {self.VERSION}). Compatibility issues may arise.")

                self.iteration = state.get("iteration", 0)
                self.error_history = deque(state.get("error_history", []), maxlen=PARAMETERS["MAX_ERROR_HISTORY"])
                self.modification_history = deque(state.get("modification_history", []), maxlen=PARAMETERS["MAX_MODIFICATION_HISTORY"])
                self.performance_history = deque(state.get("performance_history", []), maxlen=200)
                self.last_backup_time = state.get("timestamp", 0) # Use state timestamp as last backup time

                # Load components (if state exists for them)
                if state.get("space_state") and self.space:
                     self.space.load_space_state(state["space_state"])
                     # Restore next IDs from state if available, otherwise keep space's loaded IDs
                     self.space.next_neuron_id = state.get("next_neuron_id", self.space.next_neuron_id)
                     self.space.next_cluster_id = state.get("next_cluster_id", self.space.next_cluster_id)
                     self.space.next_sector_id = state.get("next_sector_id", self.space.next_sector_id)

                if state.get("knowledge_graph_state") and self.knowledge_graph:
                     self.knowledge_graph.load_state(state["knowledge_graph_state"])
                if state.get("holographic_memory_state") and self.holographic_memory:
                     self.holographic_memory.load_state(state["holographic_memory_state"])
                if state.get("genome_state") and self.genome:
                     self.genome = NebulaGenome.from_list(state["genome_state"])
                     logger.info("Loaded genome from state.")

                logger.info("Nebula state loaded successfully.")
                # Apply genome parameters AFTER loading everything
                if self.genome:
                     self.genome.apply_to_parameters()


            except EOFError:
                logger.error("Failed to load state: State file is corrupted or empty.")
                # Handle corrupted file (e.g., load backup, or start fresh)
                self._handle_corrupted_state()
            except Exception as e:
                logger.error(f"Failed to load Nebula state: {e}", exc_info=True)
                self._handle_corrupted_state() # Treat other load errors as potentially corrupt
        else:
            logger.info("No state file found. Starting Nebula with initial configuration.")


    def _handle_corrupted_state(self):
        """Attempts to load the latest backup if state file is corrupt."""
        logger.warning("Attempting to load from latest backup due to state file load error...")
        backup_dir = PARAMETERS["BACKUP_DIRECTORY"]
        try:
            backups = sorted(backup_dir.glob("nebula_state_backup_*.pkl"), key=os.path.getmtime, reverse=True)
            if backups:
                latest_backup = backups[0]
                logger.info(f"Found latest backup: {latest_backup}")
                # Try loading the backup
                with open(latest_backup, 'rb') as f:
                     state = pickle.load(f)
                 # If loading backup works, copy it to the main state file path
                shutil.copyfile(latest_backup, PARAMETERS["STATE_FILE"])
                logger.info(f"Successfully loaded state from backup and restored state file.")
                # Now call load_state again to actually load the restored data
                self.load_state()
            else:
                logger.error("No backup files found. Starting with a fresh state.")
                # Explicitly re-initialize space if needed
                if self.space: self.space.initialize_neurons(PARAMETERS["INITIAL_NEURONS"])

        except Exception as e:
            logger.error(f"Failed to load state from backup: {e}. Starting with a fresh state.", exc_info=True)
            # Ensure space is initialized if all loading fails
            if self.space: self.space.initialize_neurons(PARAMETERS["INITIAL_NEURONS"])

    # --- Monitoring & Info ---

    def get_statistics_dict(self) -> Dict:
         """Returns a dictionary of current system statistics."""
         stats = {
             "Iteration": self.iteration,
             "Uptime (s)": int(time.time() - self.start_time),
             "Neurons": len(self.space.neurons) if self.space else 0,
             "Clusters": len(self.space.clusters) if self.space else 0,
             "Sectors": len(self.space.sectors) if self.space else 0,
             "Connections (Graph)": self.space.connection_graph.number_of_edges() if self.space else 0,
             "KG Nodes": self.knowledge_graph.graph.number_of_nodes() if self.knowledge_graph else 0,
             "KG Edges": self.knowledge_graph.graph.number_of_edges() if self.knowledge_graph else 0,
             "Holo Memory Items": len(self.holographic_memory.memory) if self.holographic_memory else 0,
             "Errors Logged": len(self.error_history),
             "Modifications Applied": len(self.modification_history),
             "Avg Performance (Last 20)": f"{np.mean(list(self.performance_history)[-20:]):.3f}" if self.performance_history else "N/A",
             "RAM Usage (%)": f"{psutil.virtual_memory().percent:.1f}",
         }
         # GPU Usage (optional)
         if GPUtil_AVAILABLE and DEVICE.type == 'cuda':
              try:
                   gpus = GPUtil.getGPUs()
                   if gpus:
                       gpu = gpus[0] # Assume single GPU for simplicity
                       stats["GPU Usage (%)"] = f"{gpu.load * 100:.1f}"
                       stats["GPU Temp (°C)"] = f"{gpu.temperature:.1f}"
                       stats["VRAM Usage (%)"] = f"{gpu.memoryUtil * 100:.1f}"
                       stats["VRAM Used (GB)"] = f"{gpu.memoryUsed / 1024:.2f}"
              except Exception: stats["GPU Info"] = "Error accessing GPU" # Handle potential errors
         # Loaded LLMs
         loaded_llms = [k for k, v in self.llm_load_status.items() if v]
         stats["Loaded LLMs"] = ", ".join(loaded_llms) if loaded_llms else "None"
         return stats


    def display_statistics(self):
        """Logs current system statistics."""
        stats = self.get_statistics_dict()
        logger.info("--- Nebula Statistics ---")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        logger.info("-------------------------")
        # Also emit to UI if connected
        if PARAMETERS["UI_ENABLED"] and self.nebula_thread:
             self.nebula_thread.stats_signal.emit(stats)


    def _check_memory_usage(self, context: str, trigger_unload=True) -> bool:
        """Checks RAM and VRAM usage, logs warnings, and potentially triggers unload."""
        high_pressure = False
        try:
            # RAM
            ram_percent = psutil.virtual_memory().percent
            ram_threshold = 90.0 # PARAMETERS.get("MEMORY_PRESSURE_THRESHOLD", 90.0)
            if ram_percent > ram_threshold:
                logger.warning(f"High RAM Usage ({ram_percent:.1f}% > {ram_threshold:.1f}%) - Context: {context}")
                high_pressure = True

            # VRAM (if applicable)
            if GPUtil_AVAILABLE and DEVICE.type == 'cuda':
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    vram_percent = gpu.memoryUtil * 100
                    vram_threshold = 85.0 # PARAMETERS.get("GPU_MEMORY_THRESHOLD", 85.0)
                    if vram_percent > vram_threshold:
                         logger.warning(f"High VRAM Usage ({vram_percent:.1f}% > {vram_threshold:.1f}%) - Context: {context}")
                         high_pressure = True

            # Trigger unload if high pressure detected and allowed
            if high_pressure and trigger_unload:
                 logger.warning("High memory pressure detected. Forcing unload of inactive LLMs...")
                 # Unload all non-embedding models immediately
                 for model_key in list(self.llm_models.keys()):
                      if model_key != "embedding":
                           self.unload_llm_model(model_key, force=True)

            return high_pressure # Return whether high pressure was detected

        except Exception as e:
            logger.error(f"Failed to check memory usage: {e}", exc_info=False)
            return False # Assume not high pressure if check fails

    # --- Shutdown ---
    def shutdown(self):
        """Initiates graceful shutdown procedures."""
        if self.shutdown_requested: return # Prevent multiple calls
        logger.info("Initiating NebulaAGI shutdown...")
        self.shutdown_requested = True

        # Stop background thread if running
        if self.nebula_thread and self.nebula_thread.isRunning():
             logger.info("Waiting for Nebula thread to finish...")
             self.nebula_thread.quit()
             self.nebula_thread.wait(5000) # Wait up to 5 seconds

        # Save final state
        logger.info("Performing final state save...")
        self.save_state()

        # Release resources (e.g., unload LLMs)
        logger.info("Unloading all LLMs...")
        for model_key in list(self.llm_models.keys()):
             self.unload_llm_model(model_key, force=True)

        # Close UI if running
        if self.app:
             logger.info("Closing UI...")
             self.app.quit()

        logger.info("👋 NebulaAGI Shutdown Complete.")


# ========================================================
# UI Components (Placeholder - Requires separate nebula_ui.py)
# ========================================================
if PYQT_AVAILABLE:
    print("🎨 Defining UI Components (Requires nebula_ui.py)...")
    # Assume UserInterface and NebulaThread are defined in 'nebula_ui.py'
    # This prevents cluttering the main file and allows UI to be optional.
    try:
        from nebula_ui import UserInterface, NebulaThread
        logger.info("UI components imported from nebula_ui.py")
    except ImportError:
        logger.warning("Could not import UI classes from nebula_ui.py. UI will be disabled.")
        PARAMETERS["UI_ENABLED"] = False
        # Define dummy classes if import fails but PYQT_AVAILABLE was True initially
        class UserInterface(QMainWindow): pass
        class NebulaThread(QThread): pass

# ========================================================
# MAIN EXECUTION BLOCK
# ========================================================
if __name__ == "__main__":
    print("🚀 Starting NebulaAGI Main Execution...")

    # --- Dependency Checks ---
    if not TRANSFORMERS_AVAILABLE:
        print("\nCRITICAL ERROR: 'transformers' and 'sentence-transformers' libraries are required.")
        print("Please install them: pip install transformers sentence-transformers huggingface_hub torch")
        sys.exit(1)
    if not DEAP_AVAILABLE:
        print("\nWARNING: 'deap' library not found. Evolutionary features will be disabled.")
        print("Install if needed: pip install deap")
    if not NLP_AVAILABLE:
         print("\nWARNING: 'spacy' library not found or model 'en_core_web_sm' missing.")
         print("Some NLP features will be limited. Install: pip install spacy && python -m spacy download en_core_web_sm")

    nebula_instance = None
    try:
        # --- Hugging Face Login (Optional but Recommended) ---
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            print("Attempting Hugging Face login...")
            try:
                login(token=hf_token, add_to_git_credential=False)
                user = whoami()
                print(f"✅ Logged in to Hugging Face as: {user['name']}")
            except Exception as hf_e:
                print(f"⚠️ Hugging Face login failed: {hf_e}. Private models may be inaccessible.")
        else:
            print("⚠️ Hugging Face token not found (set HUGGINGFACE_TOKEN env var). Access may be limited.")

        # --- Instantiate and Run ---
        nebula_instance = NebulaAGI()
        nebula_instance.run()

    except KeyboardInterrupt:
        print("\n📉 NebulaAGI stopped by user (KeyboardInterrupt).")
        if nebula_instance:
             nebula_instance.shutdown()
    except Exception as main_e:
         logger.critical("💥 UNHANDLED CRITICAL EXCEPTION IN MAIN BLOCK:", exc_info=True)
         print(f"\n💥 CRITICAL ERROR: {main_e}")
         traceback.print_exc()
         if nebula_instance:
              print("Attempting emergency shutdown...")
              nebula_instance.shutdown()
    finally:
         # Optional: Any final cleanup
         if RAY_ENABLED and ray.is_initialized():
              print("Shutting down Ray...")
              ray.shutdown()
         print("\nNebulaAGI process finished.")
         sys.exit(0) # Ensure clean exit

else:
     # This allows importing classes/functions if needed, without running the main block
     logger.info("NebulaAGI module imported, not executed directly.")
