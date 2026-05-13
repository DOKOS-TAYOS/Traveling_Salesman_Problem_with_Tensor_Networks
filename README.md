# 🧭 Tensor Network TSP Solver

A quantum-inspired algorithm for solving the Traveling Salesman Problem (TSP) using tensor networks, implemented with an interactive Streamlit web application.

![TSP Visualization](TSP_Nodes.png)

## 📖 Overview

This project implements a novel quantum-inspired algorithm for solving the Traveling Salesman Problem (TSP) using tensor networks. The approach leverages quantum computing concepts like superposition and imaginary time evolution to find optimal routes efficiently.

**Author:** Alejandro Mata Ali  
**Paper:** [Traveling Salesman Problem from a Tensor Networks Perspective](https://arxiv.org/abs/2311.14344)  
**Citation:** Alejandro Mata Ali, Iñigo Perez Delgado, Aitor Moreno Fdez. de Leceta. "Traveling Salesman Problem from a Tensor Networks Perspective" arXiv:2311.14344 [quant-ph] (2023)

## 🎯 The Traveling Salesman Problem

The TSP is one of the most famous optimization challenges in computer science. Given a set of cities and distances between them, the goal is to find the shortest possible route that visits each city exactly once and returns to the starting city.

**Mathematical Formulation:**

Minimize: $C(\vec{x}) = \sum_{t=0}^{n-1} T_{x_t, x_{t+1}}$


Where:
- $x_t$ is the city visited at time step t
- $T_{i,j}$ is the distance between cities i and j  
- $x_n = x_0$ (return to starting city)

**Constraints:**
- Each city must be visited exactly once
- The tour must form a complete cycle

## ⚛️ Tensor Network Algorithm

![Tensor Network Structure](TensorNetworkTSP.png)

The algorithm implements a quantum-inspired approach consisting of four main steps:

### 1. 🔄 Quantum Superposition Initialization
Creates uniform superposition vectors representing all possible city combinations simultaneously using tensor networks.

```python
# Creates uniform superposition of all possible routes
uniform_node = tk.Node(tensor=torch.ones(n_nodes), name='uniform')
```

### 2. 🌀 Imaginary Time Evolution  
Applies evolution operators that exponentially favor shorter routes through imaginary time evolution with parameter τ (tau).

```python
# Evolution tensor favoring shorter distances
evolution_tensor = torch.exp(-tau * distances)
```

### 3. 🛡️ Constraint Enforcement
Uses restriction layers to ensure each city is visited exactly once. More layers provide stronger constraints but slower computation.

### 4. 🎯 Solution Extraction
Extracts the optimal route from the quantum state through projection and partial trace operations.

### Key Parameters:
- **τ (Tau)**: Controls optimization intensity. Higher values favor shorter routes but risk numerical overflow
- **Restriction Layers**: Number of constraint layers. `n_layers=None` applies all restriction layers used by the current solver implementation.
- **Bond Dimension**: Internal tensor-network dimension used by the implementation.

## 🖥️ Streamlit Web Application

An interactive web interface that makes the quantum TSP solver accessible to everyone.

### ✨ Features

#### 📊 Problem Configuration
- **Manual Entry**: Interactive distance matrix editor
- **File Import**: Support for CSV and JSON formats with automatic size detection
- **Random Generation**: Create test problems with configurable parameters
- **Problem Validation**: Real-time checks for matrix completeness

#### 🎛️ Algorithm Controls
- **City Count**: Configure problem size (5-40 cities)
- **Parameter Tuning**: Adjust τ and restriction layers
- **Solver Options**: Customizable algorithm settings

#### 📈 Results Visualization
- **Performance Metrics**: Runtime and solution quality analysis
- **Interactive Graph**: Network visualization with optimal route highlighting
- **Step-by-Step Route**: Detailed path breakdown with distances
- **Solution Export**: Download results as JSON with complete metadata

#### 💡 User Guidance
- **Real-time Tips**: Context-aware troubleshooting suggestions
- **Parameter Help**: Detailed explanations for algorithm settings
- **Example Files**: Pre-configured test cases for learning

### 🎨 Interface Highlights
- **Responsive Design**: Works on desktop and mobile devices
- **Dark Theme Support**: Modern, professional appearance
- **Real-time Feedback**: Instant validation and status updates
- **Progressive Disclosure**: Advanced options available when needed

## 🚀 Installation & Usage

### Prerequisites
```bash
Python 3.11 or higher
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Install Benchmark Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-benchmark.txt
```

### Launch the Application
```bash
streamlit run streamlit_app.py
```

The web application will open in your browser at `http://localhost:8501`

### Command Line Usage
```python
from tsp_module import tn_tsp_solver
import torch

# Define distance matrix
distances = torch.tensor([[0, 10, 15, 20],
                         [10, 0, 12, 18], 
                         [15, 12, 0, 14],
                         [20, 18, 14, 0]], dtype=torch.float32)

# Solve TSP
solution = tn_tsp_solver(distances, tau=1.0, n_layers=3)
print(f"Optimal route: {solution}")
```

## 📁 Project Structure

```
TSP/
├── 📄 streamlit_app.py          # Main Streamlit web application
├── 🧠 tsp_module.py             # Core tensor network TSP solver
├── 🔧 auxiliary_functions.py    # Helper functions for problem generation
├── 📥 import_functions.py       # File import/export functionality
├── 🖼️ TSP_Nodes.png            # TSP problem visualization
├── 🖼️ TensorNetworkTSP.png     # Algorithm structure diagram
├── 📊 tn_tsp_tests.ipynb        # Jupyter notebook with examples
├── 📋 requirements.txt          # Python dependencies
├── 📄 paper.pdf                 # Original research paper
├── 🧪 test_functions.py         # Unit tests
├── 🔍 auxiliary_solver.py       # Additional solver utilities
└── 📂 example_*.{csv,json}      # Sample input files
```

## 🔬 Technical Implementation

### Core Components

#### Tensor Network Construction
- **Superposition Layer**: Creates uniform distribution over all possible routes
- **Evolution Layer**: MPO tensors implementing imaginary time evolution
- **Restriction Layer**: Enforces TSP constraints through projection operators
- **Contraction**: Efficient tensor network contraction for solution extraction

#### Optimization Features
- **Template Tensors**: Memory-efficient implementation using shared tensor templates
- **Progressive Refinement**: Iterative improvement through multiple restriction layers
- **Numerical Stability**: Automatic overflow detection and parameter adjustment

#### Implementation Notes
- **Tensor Construction**: Uses the tensor-network layers implemented in `tsp_module.py`
- **PyTorch Operations**: Uses PyTorch tensors for matrix and tensor operations
- **Restriction Layers**: Supports exact runs and reduced-layer heuristic runs through `n_layers`

### Supported Input Formats

#### CSV Format
```csv
,City 0,City 1,City 2
City 0,0,10,15
City 1,10,0,20  
City 2,15,20,0
```

#### JSON Format (Simple)
```json
[[0,10,15],[10,0,20],[15,20,0]]
```

#### JSON Format (Structured)
```json
{
  "distance_matrix": [[0,10,15],[10,0,20],[15,20,0]],
  "problem_name": "Example TSP",
  "city_names": ["City A", "City B", "City C"]
}
```

## 📊 Algorithm Performance

### Complexity Analysis
- **Time Complexity**: O(n² × 2^n × D²) where D is bond dimension
- **Space Complexity**: O(n × D²)
- **Approximation Quality**: Controllable through restriction layers and bond dimension

### Scalability
- **Small Problems** (≤10 cities): Exact solutions possible
- **Medium Problems** (10-25 cities): High-quality approximations
- **Large Problems** (25+ cities): Heuristic solutions with adjustable quality

### Benchmarking
The benchmark suite is intended to validate correctness on small TSP instances and to characterize the behavior of the layer-removal heuristic. It is not intended to establish computational superiority over specialized classical solvers.

Smoke test:
```bash
pytest -q
python -m benchmarks.run_tsp_benchmarks --suite small_exact --dry-run
```

Small exact experiment:
```bash
python -m benchmarks.run_tsp_benchmarks --suite small_exact --out results/tsp_small_exact.csv --sizes 5 6 7 8 9 10 --seeds 0 1 2 3 4 --tau 1.0
```

Tau calibration experiment:
```bash
python -m benchmarks.run_tau_sweep --out results/tau_sweep.csv --summary-out results/tau_by_size.csv --sizes 5 6 7 8 9 10 12 --seeds 0 1 2 3 4 --taus 0.05 0.1 0.2 0.5 1 2 5 10 20 50
```

Use calibrated tau values in later benchmarks:
```bash
python -m benchmarks.run_tsp_benchmarks --suite small_exact --out results/tsp_small_exact_tau_calibrated.csv --sizes 5 6 7 8 9 10 --seeds 0 1 2 3 4 --tau-by-size results/tau_by_size.csv
```

Layer-removal experiment:
```bash
python -m benchmarks.run_layer_ablation --out results/layer_ablation.csv --sizes 8 10 12 --seeds 0 1 2 3 4 --layer-ratios 0.0 0.1 0.25 0.5 0.75 1.0 --repeats 25 --tau 1.0
```

Plots and summary:
```bash
python -m benchmarks.make_plots --results-dir results --out-dir reports/figures
```

Publication-style figures for LaTeX:
```bash
python -m benchmarks.paper_figures --results-dir results --out-dir reports/paper_figures
```

The purpose of these benchmarks is to validate correctness on small TSP instances and to characterize the behavior of the layer-removal heuristic. They are not intended to establish computational superiority over specialized classical solvers.

## 🛠️ Configuration Options

### Algorithm Parameters
| Parameter | Range | Description | Impact |
|-----------|-------|-------------|---------|
| τ (tau) | 0.1-10.0 | Evolution strength | Higher = prefer shorter routes |
| Layers | 1 to n-1 | Restriction layers | More = better constraints |
| Bond Dim | Auto | Tensor bond dimension | Higher = more accuracy |

### Problem Parameters  
| Parameter | Range | Description |
|-----------|-------|-------------|
| Cities | 5-40 | Problem size |
| Connections | 2 to n-1 | Max connections per city |
| Distance Range | 1-100 | Random distance bounds |

## 🔧 Troubleshooting

### Common Issues

**Sequential Solution (0,1,2,3...)**
- Solution: Reduce τ value
- Cause: Evolution too strong, favoring trivial path

**Repeated Cities in Solution**  
- Solution: Increase restriction layers
- Cause: Insufficient constraint enforcement

**Numerical Overflow**
- Solution: Reduce τ parameter
- Cause: Exponential evolution terms too large

**Slow Performance**
- Solution: Reduce problem size or layers
- Cause: Exponential scaling with problem size

### Performance Tips
- Start with small problems (5-7 cities) to understand behavior
- τ = 1.0 is usually a good starting point
- More layers = better quality but slower speed
- Symmetric distance matrices generally work better

## 📚 References & Citation

If you use this software in your research, please cite the original paper:

```bibtex
@article{mata2023traveling,
  title={Traveling Salesman Problem from a Tensor Networks Perspective},
  author={Mata Ali, Alejandro and Perez Delgado, I{\~n}igo and Moreno Fdez. de Leceta, Aitor},
  journal={arXiv preprint arXiv:2311.14344},
  year={2023}
}
```

**Paper URL:** https://arxiv.org/abs/2311.14344

## 📞 Contact & Support

**Author:** Alejandro Mata Ali  
**Institution:** [Based on paper affiliations]

For questions about the algorithm or implementation, please refer to the original paper or create an issue in this repository.

## 📄 License

Please refer to the original paper for licensing information regarding the algorithm. The Streamlit implementation follows standard open-source practices.

---

*This implementation brings cutting-edge quantum-inspired optimization to your fingertips through an intuitive web interface. Explore the fascinating world of tensor networks and quantum computing applied to classical optimization problems!* 🚀✨
