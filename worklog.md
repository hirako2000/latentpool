# 📓 Development Worklog

## Session 1: Setup

_2026-01-30_

### ✅ Done

The project already has a solid set up.

- **Environment Setup:** Initialized the project using `uv` with Python 3.14. Configured `pyproject.toml` with core dependencies (`torch`, `torch-geometric`, `msgspec`, `httpx`).
- **Project Structure:** Established the `src/latentpool` layout and defined the initial CLI entry points using `typer`.
- **Provider Implementation:** Created the `NodeProvider` class to handle JSON-RPC communication with Ethereum nodes, specifically targeting `trace_transaction` and `trace_block`.
- **GNN Scaffolding:** Implemented a base `TransactionGraphModel` using `GraphSAGE` layers to define the expected input/output tensor shapes.
- **Initial Testing:** Set up `pytest` and `respx` for mocking RPC calls. Created `test_provider.py` to verify data ingestion logic.
- **Branding:** Created the `README.md` with the project vision, hero image, and feature roadmap.
- **pytest/ruff**: making sure to have those working from the get go.

### 🧠 Challenges

I don't know what to say, Python still feels sneaky. Nothing took too long to address.

- **Python 3.14 Compatibility:** Navigated the early-stage support for 3.14, ensuring `uv` correctly managed the experimental toolchain.
- **JSON-RPC Complexity:** Deciphered some parity trace format to ensure the `msgspec` structs correctly mapped hex strings to obj.
- **Mocking Strategy:** Decided on `respx` for HTTP mocking early on to avoid hitting real nodes during the initial phases.

### 🚧 Struggles
- **Dependency Hell:** Balancing the specific versions of `torch` and `torch-geometric` that play nice, but then figured it didn't like python 3.13. Moved to 3.14 

### 🎯 Status

- Basic "Hello World" CLI to see things running.
- Able to fetch and decode transaction traces in a test environment.
- GNN architecture drafted but not yet trained.

### Next?

Static code analysis, improve test coverage, get GPU accessleration (metal/MPS).

## Session 2: Some hardening

_2026-01-30_

### ✅ Done
- **Static Analysis:** Integrated `pyright` in `strict` mode.
- **Testing:** Achieved 100% code coverage across `gnn.py` and `provider.py`.
- **Hardware Abstraction:** Implemented and tested conditional device routing (MPS/CUDA/CPU).
- **Orchestration:** Integrated `justfile` for a unified dev workflow.
- **Refactoring:** Cleaned up `provider.py` with `typing.cast` and `msgspec` optimized decoders.

### 🧠 Challenges
- **Strict Typing vs. Libraries:** Encountered `reportMissingTypeStubs` with `torch_geometric`. Went with a balance of strictness by silencing noisy library stubs while keeping internal logic tight.
- **Hardware Mocking:** Faced `AssertionError` when mocking CUDA availability. Learned that PyTorch's "Lazy Init" requires mocking `__init__` or using `unittest.mock.patch` to test hardware-logic without hardware drivers.
- **Msgspec Factory:** Debugged `default_factory=list` in `msgspec.Struct`, resolved by using `default=[]` to satisfy Pyright's generic inference.

### 🚧 Struggles

None, growing number of commands, solved by integrating a [justfile](https://github.com/casey/just)

### 🎯 Current Status

100% Coverage, 0 Linting errors, 0 Type errors.

### Next?

May focus on `httpx` performance (connection pooling) and graph construction logic. Perhaps set up a benchmarking toolchain, as there will be much more to measure.


## Session 3: Benchmarking facility 

_2026-01-31_

pytest has a benchmark extension, experimenting...

### ✅ Done 

Integrated, now running the check or tests will run the benchmark, default to 100 rounds for warm up and decent min/max median figure.

- Optimized request as http handshakes are expensive. About 30% improvement on the ingestion benchmark just with that.
- tried to optimize the json parsing, but it's only saving a few microseconds and make the code terribly ugly. So I gave up.

### 🧠 Challenges

None, except insisting to make some marginal performacne improvement on json parsing, was not worth it. 

### 🚧 Struggles

Unit testing code smell is very hard. Luckily I did not have to fix the tests as I dumped the optimization as unecessary

### 🎯 Current Status

Still 100% code coverage, now with benchmark, less than 1ms spent ingesting what will look like typical transactions from real mem pools.

### Next?

I don't know. I guess it's time to make progress on the GNN, data engineering.