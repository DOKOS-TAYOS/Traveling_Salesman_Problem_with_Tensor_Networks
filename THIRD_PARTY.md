# Third-party notices

This project is released under the MIT License (see [LICENSE](LICENSE)).
The notices below cover third-party code and dependencies used by this
repository.

## Derived source code

### Google OR-Tools TSP tutorial sample

- File: [`auxiliary_solver.py`](auxiliary_solver.py)
- Source: [Traveling Salesperson Problem (OR-Tools)](https://developers.google.com/optimization/routing/tsp)
- Copyright: Google LLC
- License: [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

`auxiliary_solver.py` adapts the routing-model setup, distance callback, search
parameters, and solution extraction patterns from that sample for use with
PyTorch distance tensors in this repository.

## Main Python package dependencies

Licenses below are the typical upstream licenses for the packages as commonly
distributed on PyPI. Confirm the license shipped with the exact version you
install if you need a formal compliance audit.

| Package | Typical license | Role in this project |
| --- | --- | --- |
| [ortools](https://pypi.org/project/ortools/) | Apache-2.0 | Classical TSP baseline (`auxiliary_solver.py`, benchmarks) |
| [torch](https://pypi.org/project/torch/) | BSD-style (PyTorch) | Tensor operations for the TN solver |
| [tensorkrowch](https://pypi.org/project/tensorkrowch/) | See package metadata | Tensor-network construction |
| [numpy](https://pypi.org/project/numpy/) | BSD-3-Clause | Numerical arrays |
| [pandas](https://pypi.org/project/pandas/) | BSD-3-Clause | Data handling / CSV I/O |
| [matplotlib](https://pypi.org/project/matplotlib/) | PSF-based / matplotlib | Plots and figures |
| [networkx](https://pypi.org/project/networkx/) | BSD-3-Clause | Graph heuristics / visualization |
| [streamlit](https://pypi.org/project/streamlit/) | Apache-2.0 | Interactive web app |
| [ipython](https://pypi.org/project/ipython/) | BSD-3-Clause | Notebook / interactive use |
| [scipy](https://pypi.org/project/scipy/) | BSD-3-Clause | Benchmark extras |
| [pytest](https://pypi.org/project/pytest/) | MIT | Test runner |
| [psutil](https://pypi.org/project/psutil/) | BSD-3-Clause | Benchmark memory metrics |
| [python-tsp](https://pypi.org/project/python-tsp/) | MIT (optional) | Optional classical wrappers |
| [ruff](https://pypi.org/project/ruff/) | MIT | Lint (benchmark requirements) |

Runtime pins are listed in `requirements.txt` and `requirements-use.txt`.
Benchmark and test extras are listed in `requirements-benchmark.txt`.
