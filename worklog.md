## 📓 Development Worklog

### Foundation
_2026-01-29_: Session 1

Setting up the project with `uv`. The design will grow, starting with a `NodeProvider` utilizing `msgspec` for JSON-RPC decoding. For processing Ethereum traces at scale without the performance penalties typical of standard JSON libraries.

Defined Parity-style structures using `msgspec.Struct`. This provides a speed advantage over Pydantic for the upcoming real-time mempool data.
**GNN Scaffolding:** Initialized the `TransactionGraphModel` with GraphSAGE. Will deal with GPU later.

###  System Hardening

_2026-01-30_: Session 2

Developer UX and reliability. Added `pyright` in strict mode to catch poorly defined type early. Managing strict types alongside libraries like `torch_geometric` required a balanced approach: silencing external library stubs while maintaining high internal standards.

Refactored device selection logic to be fully testable. Using `unittest.mock.patch` allowed me to verify hardware routing without needing specific physical drivers during CI.
Added a `justfile` to consolidate commands. This abstracts away the verbosity of the `uv` toolchain.

Design so far:

```text
    [ Ethereum Node / Archive API ]
           |
           | (JSON-RPC: trace_block)
           v
    [ NodeProvider ] ----------------> [ msgspec Decoder ]
           |                                  |
           | (Persistent Async Connection)    | (Strict Type Mapping)
           v                                  v
    [ Feature Vectorizer ] <---------- [ TransactionTrace ]
           |
           | (Graph Construction)
           v
    [ torch_geometric.Data ]
           |
           | (Tensor Routing: MPS/CUDA/CPU)
           v
    [ TransactionGraphModel ] -------> [ MEV Prediction ]
```

### Perf Profiling

_2026-01-31_: Session 3

Need to measure latency. Added `pytest-benchmark`, already seeing some overhead with the HTTP handshake overhead rather than data parsing.


Added **Connection Pooling**. Refactored the provider to use a persistent `httpx.AsyncClient`. This change reduced ingestion latency by approximately 30% by reusing TCP connections.

I explored deep micro-optimizations for JSON parsing but ultimately discarded them. The marginal gains did not justify the increased code complexity.

_Profiling aspects:

```text
[ Ethereum Node / Archive API ]
           |
           | (JSON-RPC: trace_block)
           |  Latency: ~30% reduction via connection pooling
           v
    [ NodeProvider ] ----------------> [ msgspec Decoder ]
           |                                  |
           | (Persistent AsyncClient)         | (Pre-compiled Type Decoder)
           |  Handshake reuse: OK             | Parsing: < 1ms overhead
           v                                  v
    [ Feature Vectorizer ] <---------- [ TransactionTrace ]
           |
           | (Graph Construction)
           v
    [ torch_geometric.Data ]
           |
           | (Hardware Routing)
           |  Auto-dispatch: MPS (Metal) or CPU
           v
    [ TransactionGraphModel ] -------> [ MEV Prediction ]
           |
           | (GraphSAGE Inference)
           |  Warm-up: 100 rounds\
```

### dvc and hugging face

_2026-02-02_: Session 4

There is the need to track datasets, and host them. Opting for DVC for versioningr and huggingface for hosting datasets, and the model weights.

Also spent some times revisiting the current doc, added separate documents for the dvc and huggingface aspects, also a good overview on the data engineering steps intended for reproducibility and traceability. Mentioned xAI.

Aded unit tests to cover the boilerplate cli, added some preliminary code for pulling block info from some archive done
Alsy a hydrator class to get started with data processing.

I didn't test any of these, added a full suite though to validate the logic is correct

Now have a data_config.py file, managed .env file as I will need to hook the data preparation process with Alchemy or infura.

Added a CI workflow to perform static code analysis and run all the tests in a separate environment.

### Silver analysis and 'normal' transactions

*2026-02-03*: Session 5

The hardest sessions, it's in fact two. I had worked on this yesterday until late.
and all day today...

The hydration logic worked but was missing non attack transactions. The training data needs more than just MEVs; Added in "Normal" transaction samples to serve as negative labels, ensuring the model doesn't just learn to flag every high-value swap.

Revised the `IngestionCoordinator` to perform **Negative Sampling**. The hydrator now fetches MEV receipts from labeled CSVs and immediately samples "sibling" transactions from the same blocks. This creates a balanced 1:1 ratio for the 3-class classification (Arbitrage vs. Sandwich vs. Normal).

Built a suite of visualizations to validate the Silver layer before it hits the Gold (labeled) joiner.

added a document for these: [silver-viz.md](./docs/silver-viz.md)

Also Refactored the CLI into a nested hierarchy (`data`, `viz`, `silver`) to improve DX.  100% code coverage is a pain, but it helped me catch a few issues.

_Data Pipeline_

```text
    [ Labeled CSVs (MEV) ]
           |
           v
    [ IngestionCoordinator ] <------- [ Alchemy / Archive ]
           |                              |
           | (Step 1: MEV Hydration)      | (Step 2: Negative Sampling)
           |  Target: 1:1 Ratio           |  Source: Sibling Blocks
           v                              v
    [ Silver Layer (edges.parquet) ] <------- [ Diagnostics ]
           |                                     |-- Structure (Outliers)
           | (Join & Label)                      |-- Temporal (Density)
           v                                     |-- Diversity (Hubs)
    [ Gold Training Set ] <------- TODO
           |
           | (Graph Construction) <------- TODO
           v
    [ PyG Tensors / Graphs ] <------- TODO
```
