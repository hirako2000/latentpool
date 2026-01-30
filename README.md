# 🌊 LatentPool

<div align="center">
<img src="./img/logo.avif" alt="LatentPool Hero" width="50%">
  <br />
  <b>Hidden MEV Patterns in Mempools with Graph Neural Networks</b>
</div>

**Hidden MEV Patterns in Mempools with Graph Neural Networks**

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v0.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LatentPool detects Maximum Extractable Value (MEV) by analyzing transaction graphs with Graph Neural Networks (GNNs). It identifies complex cross-contract patterns without requiring bytecode analysis or ABIs.

## 📑 Table of Contents
- [✨ Features](#features)
- [🚀 Quick Start](#quick-start)
- [🛠 How It Works](#how-it-works)
- [📊 Performance Benchmark](#performance-benchmark)
- [🏗 Project Structure](#project-structure)
- [🥶 TODOs](#todos)
- [🧪 Development](#development)
- [📜 License](#license)

---

## ✨ Features

* **⚡ ABI-Free Detection:** Identifies MEV via latent graph topology—no contract knowledge required.
* **🧠 Advanced GNNs:** Built-in support for **GraphSAGE**, **GAT**, and **GCN** architectures.
* **🚀 Modern Engine:** Powered by `uv` for lightning-fast environment management and `Ruff` for linting.
* **🍎 Apple Silicon Optimized:** Native **MPS (Metal Performance Shaders)** acceleration for M-series.
* **📦 Grade:** Typed with `msgspec` for high-performance JS decoding and `pytest` for reliability, but experimental project.

---

## 🚀 Quick Start

### Installation
The project uses [uv](https://github.com/astral-sh/uv) for the dev experience, this also needs python version >= 3.14

```bash
# Clone the repository
git clone https://github.com/hirako2000/latentpool
cd latentpool

# Sync environment (installs Python 3.14 & all dependencies)
uv sync

```

### Usage

```bash
# Detect MEV in a specific block
uv run latent detect

# Monitor mempool in real-time ?
see TODOs further down
```

---

## 🛠 How It Works

LatentPool bypasses traditional simulation-based detection by treating the mempool as a dynamic graph.

1. Graph Construction: Transactions are converted into directed graphs of state changes.
2. Feature Vectorization:** Nodes are embedded with 14 key features capturing value flow.
3. GNN Inference: GraphSAGE layers aggregate neighborhood information to flag suspicious clusters.
4. Hardware Acceleration: Tensors are automatically routed to **MPS (Metal)** (for now) or CUDA for sub-max latency.

# TODOs

Still a lot left to _do_.

### Dev setup and UX 

- [x] All UV
- [x] Pytest with coverage report
- [x] justfile for commands convenience

### Quality Gates

Using a somewhat _Strict_ toolchain to ensure code reliability:
[x] Linting: `uv run ruff check` (Logical & Security analysis)
[x] Static code analysis: `uv run pyright` (complexity and other logical checks)
[x] Testing: `uv run pytest`

100% coverage is a target, not a goal. Pyright set to Strict-mode, will see how long it lasts but so far it is useful. Disabled stub types check, even pytorch cannot do types right.

## 📊 Performance Benchmark progress

- [ ] **Ingestion Latency:** Benchmark `msgspec`?
- [ ] **Graph Construction:** Profile CPU overhead of `torch_geometric.data.Data` creation.
- [ ] **Inference Speed:** Track targets on MPS & CUDA, vs CPU.
- [ ] **End-to-End:** Real-time mempool-to-prediction total latency.

### 🧪 Core Development progress and experimentations
- [ ] Connection Pooling: Refactor `NodeProvider` to use a persistent `httpx.AsyncClient`.
- [ ] Optimization: Could use `msgspec.Raw` to avoid double-encoding in RPC execution.
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
```

---

## 🧪 HOWTO Development

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

## 📜 License

**MIT License**. See `LICENSE` for more information.


**Disclaimer:** *LatentPool is an experimental tool for MEV research. Use at your own risk.*