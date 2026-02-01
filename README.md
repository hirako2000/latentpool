# 🌊 LatentPool

<div align="center">
<img src="./img/logo.avif" alt="LatentPool Hero" width="50%">
  <br />
  <b>Hidden MEV Patterns in Mempools with Graph Neural Networks</b>
</div>

<br/>

LatentPool detects Maximum Extractable Value (MEV) by analyzing transaction graphs with Graph Neural Networks (GNNs). It identifies complex cross-contract patterns without requiring bytecode analysis or ABIs.

<!-- [![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-%233C2179.svg?style=for-the-badge&logo=pyg&logoColor=white)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=for-the-badge)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v0.json&style=for-the-badge)](https://github.com/astral-sh/ruff)
<br/>
[![Codacy Badge](https://img.shields.io/codacy/grade/6f5fd50cc2d2480b876efa6d8223c71e?style=for-the-badge)](https://app.codacy.com/gh/hirako2000/latentpool/dashboard)
[![Bandit Security Scan](https://img.shields.io/github/actions/workflow/status/hirako2000/latentpool/bandit.yml?branch=master&label=CI&style=for-the-badge)](https://github.com/hirako2000/latentpool/actions/workflows/bandit.yml)
[![security: bandit](https://img.shields.io/badge/security-bandit-green.svg?style=for-the-badge)](https://github.com/PyCQA/bandit)
![Coverage](https://raw.githubusercontent.com/hirako2000/latentpool/badges/coverage.svg?style=for-the-badge)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT) -->

<div align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.14+-blue.svg?style=for-the-badge" alt="Python 3.14+">
  </a>
  <a href="https://pytorch.org/">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://pytorch-geometric.readthedocs.io/">
  <img src="https://img.shields.io/badge/PyTorch%20Geometric-%233C2179.svg?style=for-the-badge&logo=pyg&logoColor=white" alt="PyTorch Geometric">
  </a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=for-the-badge" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v0.json&style=for-the-badge" alt="Ruff">
  </a>
</div>

<div align="center">
  <a href="https://app.codacy.com/gh/hirako2000/latentpool/dashboard"><img src="https://img.shields.io/codacy/grade/6f5fd50cc2d2480b876efa6d8223c71e?style=for-the-badge" alt="Codacy Grade"></a>
  <a href="https://github.com/hirako2000/latentpool/actions/workflows/bandit.yml"><img src="https://img.shields.io/github/actions/workflow/status/hirako2000/latentpool/bandit.yml?branch=master&label=CI&style=for-the-badge" alt="CI Status"></a>
  <!-- <img src="https://raw.githubusercontent.com/hirako2000/latentpool/badges/coverage.svg" alt="Coverage">-->
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License: MIT"></a>
</div>


## 📑 Table of Contents
- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [🛠 How It Works](#-how-it-works)
- [📊 Performance Benchmark](#-performance-benchmark)
- [🏗 Project Structure](#-project-structure)
- [🥶 TODOs](#🥶-todos)
- [💻 Development](#-howto-development)
- [🤓 Worklog](#-worklog)
- [⚖️ License](#-license)

---

## ✨ Features

* Identifies MEV via latent graph topology.
* **🧠 GNNs:** Built-in support for [GraphSAGE](https://snap.stanford.edu/graphsage/), [GAT](https://arxiv.org/abs/1710.10903), and **GCN** architectures.
* **🚀 Modern dev flow:** Powered by `uv` for lightning-fast environment management and `Ruff` for linting.
* **🍎 Apple Silicon Optimized:** Native **MPS (Metal Performance Shaders)** acceleration for M-series. CUDA support planned.
* Typed with `msgspec` for high-performance JS decoding and `pytest` for reliability, but experimental project.

## 🚀 Quick Start

### Installation
The project uses [uv](https://github.com/astral-sh/uv) for the dev experience, this also needs python version >= 3.14

```bash
# Clone the repository
git clone https://github.com/hirako2000/latentpool
cd latentpool

# Sync environment (installs all dependencies)
uv sync
```

### Usage

```bash
# Detect MEV in a specific block
uv run latent detect

# Monitor mempool in real-time ?
see TODOs further down
```

## 🛠 How It Works

LatentPool bypasses traditional simulation-based detection by treating the mempool as a dynamic graph.

1. Graph Construction: Transactions are converted into directed graphs of state changes.
2. Feature Vectorization:** Nodes are embedded with 14 key features capturing value flow.
3. GNN Inference: GraphSAGE layers aggregate neighborhood information to flag suspicious clusters.
4. Hardware Acceleration: Tensors are automatically routed to **MPS (Metal)** (for now) or CUDA for sub-max latency.

# 🥶 TODOs

Still a lot left to _do_.

### Dev setup and UX 

- [x] All UV
- [x] Pytest with coverage report
- [x] Static code analysis
- [x] justfile for commands convenience
- [x] Run benchmarks

### Quality Gates

Using a somewhat _Strict_ toolchain to ensure code reliability:
- [x] Linting: `uv run ruff check` (Logical & Security analysis)
- [x] Static code analysis: `uv run pyright` (complexity and other logical checks)
- [x] Testing: `uv run pytest`

100% coverage is a target, not a goal. Pyright set to Strict-mode, will see how long it lasts but so far it is useful. Disabled stub types check, even pytorch cannot do types right.

### 📊 Performance Benchmark

- [x] **Ingestion Latency:** Benchmark `msgspec`
- [x] **Multi round benchmark** to ensure warm up
- [ ] **Graph Construction:** Profile CPU overhead of `torch_geometric.data.Data` creation.
- [ ] **Inference Speed:** Track targets on MPS & CUDA, vs CPU.
- [ ] **End-to-End:** Real-time mempool-to-prediction total latency.

### 🧪 Core Development progress and experimentations

- [x] Connection Pooling: Refactor `NodeProvider` to use a persistent `httpx.AsyncClient`.
- [x] Optimization: Could use `msgspec.Raw` to avoid double-encoding in RPC execution.
- [ ] Feature Engineering: attempt to implement the 14-key feature vectorization logic in some `core/` package.
- [ ] Real-time Monitoring: Implement the `latent monitor` command.

## 🏗 Project Structure

provisional...

```bash
src/latentpool/
├── cli.py         # Typer-based Command Line Interface
├── provider.py    # Execution layer node
├── gnn.py         # PyTorch Geometric model architectures
├── schema.py      # High-performance msgspec data models
├── core/          # Graph construction & feature logic
└── data/          # Blockchain data ingestion
tests/
├── test_%%%.py.   # unit test
└── test_etc.py.   # etc...
```

## 🧠 Training

Uses reproducible training pipeline.

#### Data and model Hosting

Datasets are versioned using **DVC** and hosted alongside our model weights on **Hugging Face**. Detailed instructions on dataset pulls and model uploads can be found in the [Hugging Face Guide](./docs/dvc-huggingface.md).
* **Execution:**
```bash
# Execute a training run
uv run latent train --model SAGE --epochs 40
```

#### Training

Graph Construction: Transactions are converted into directed graphs of state changes. Detailed pipeline logic is documented in [Data Engineering](./docs/data-engineering.md).

## 💻 HOWTO Development

Run the tests, which would also check for linting issues.

```bash
# Run tests (includes GNN & Ruff linting)
uv run pytest

# Manual lint & format
uv run ruff check --fix .
uv run ruff format .
```

For convenience, there is a justfile. Install [just](https://github.com/casey/just), so that all relevant commands can be invoked with short hands:

```bash
just --list
Available recipes:
    analysis   # Run static type analysis
    check      # Run all checks
    default    # Help
    fix        # Fix all auto-fixable linting and formatting issues
    lint       # Check linting and formatting without fixing
    setup      # Install/Sync dependencies
    test *args # Run the test suite with coverage
```

## Contributing

Adhere to what appears to be conventional in there. Make sure the tests pass and lint fix before committing.

## 🤓 Worklog

A [worklog.md](./worklog.md) is in there,  tracked most development sessions, what was done, learned, and challanges faced.

## ⚖️ License

**MIT License**. See `LICENSE` for more information.

**Disclaimer:** *LatentPool is an experimental tool for MEV research. Use at your own risk.*