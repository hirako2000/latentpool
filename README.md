# 🌊 LatentPool

<div align="center">
<img src="./img/logo.heic" alt="LatentPool Hero" width="50%">
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
- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [🛠 How It Works](#-how-it-works)
- [📊 Performance Benchmark](#-performance-benchmark)
- [🏗 Project Structure](#-project-structure)
- [🧪 Development](#-development)
- [📜 License](#-license)

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
The project uses [uv](https://github.com/astral-sh/uv) for the dev experience, you will also need python version >= 14.

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
uv run latent detect --block 19000000

# Monitor mempool in real-time ?
TODO
```

---

## 🛠 How It Works

LatentPool bypasses traditional simulation-based detection by treating the mempool as a dynamic graph.

1. **Graph Construction:** Transactions are converted into directed graphs of state changes.
2. **Feature Vectorization:** Nodes are embedded with 14 key features capturing value flow.
3. **GNN Inference:** GraphSAGE layers aggregate neighborhood information to flag suspicious clusters.
4. **Hardware Acceleration:** Tensors are automatically routed to **MPS (Metal)** or **CUDA** for sub-20ms latency.

---

## 📊 Performance Benchmark

TODO

---

## 🏗 Project Structure

provisional...

```text
src/latentpool/
├── cli.py         # Typer-based Command Line Interface
├── gnn.py         # PyTorch Geometric model architectures
├── schema.py      # High-performance msgspec data models
├── core/          # Graph construction & feature logic
└── data/          # Blockchain data ingestion

```

---

## 🧪 Development

You can run the tests, which would also check for linting issues.

```bash
# Run tests (includes GNN & Ruff linting)
uv run pytest

# Manual lint & format
uv run ruff check --fix .
uv run ruff format .

```

## 📜 License

**MIT License**. See `LICENSE` for more information.


**Disclaimer:** *LatentPool is an experimental tool for MEV research. Use at your own risk.*