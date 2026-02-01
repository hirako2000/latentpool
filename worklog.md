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