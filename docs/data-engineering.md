

# 🏗️ Data Engineering for LatentPool

The goal is to produce a model fast enough for inference to be sub seconds, and to ensuring that every prediction made by the GNN is traceable, reproducible, and explainable.

## Medallion Architecture

Following the **Medallion Architecture** to incrementally increase data quality and structure. This ensures an _Immutable Source of Truth"_ (Bronze) while delivering _Model-Ready_ features (Gold).

### Bronze (Raw)
- Source: eth_getTransactionReceipt
- Format: Raw JSON objects containing the logs array, gas usage, and status.

Blockchain nodes are expensive/slow. We fetch once, store forever.

### Silver (Validated)
Normalized hex strings, filtered failed transactions, and standardized token decimals.
_Garbage in, garbage out_. Cleaning ensures the model learns MEV patterns, not RPC noise.

### Gold (Features)

Represents the final articfacts: Feature vectors and PyTorch Geometric `Data` objects.

## ⚡ The Trace vs. Receipt Pivot
Initially, the pipeline was designed to use debug_traceTransaction to capture internal state changes. To optimize inference latency, we opted for a Log-Based (Receipt) Ingestion model.

Rationale: eth_getTransactionReceipt is more broadly available and provides structured logs (Swaps, Transfers) which contain 90% of the signal needed for MEV detection.

Performance: Ingestion latency dropped from ~2s/tx (tracing) to <100ms/tx (receipts), enabling near real-time mempool analysis.

## Ingestion with Version Control

We use **DVC** for data versioning, along with **Hugging Face** as the remote storage.

* **Data Lineage:** Each training batch is tagged. If a model’s F1-score drops, we can revert to the exact dataset version used in the previous "Gold" run.
* **Source Attribution:** Every graph node retains a metadata pointer to its original `tx_hash` and `log_index`.
* *The xAI Aspect:* To explain *why* a transaction was flagged as "Sandwich," we must be able to "walk back" from the GNN's hidden layer to the specific raw transfer event in the Bronze layer.


## Transformation

The `Transformation` layer converts EVM state changes into a fixed-width vector with its dimension being the number of involved addresses.

| Key | Feature | Rationale |
| --- | --- | --- |
| 1-3 | **Flow Dynamics** | Net balance changes of the `mev_taker` across the transaction's log sequence. |
| 4-7 | **Event Topology** | Count of `Swap`, `Transfer`, and `Sync` events per contract node. |
| 8-11 | **Value Metrics** | Volume and "Gas-to-Value" ratio derived from receipt `gasUsed`. |
| 12-14 | **Address Metadata** | Node degree within the transaction (how many pools a contract interacted with). |


## Visualizations & Observability

The pipeline includes **"At-a-Glance"** health checks:

* **Topology Gallery:** Automated rendering of transaction graphs (NetworkX/PyVis) during the Silver-to-Gold transition.
* **Feature Heatmaps:** Real-time distribution plots of the features to detect "Dead Features" (values that never change).
* **Label Balance:** Monitoring the ratio of Arbitrage vs. Noise to prevent model bias.

## Benchmarking

To identify bottlenecks. Measure.

**Ingestion Throughput:** Transactions per second (TPS) fetched from the node.
**Transformation Latency:** Time taken to vectorize a single trace.

## Training vs Inference

Data Engineering meets inference aspects of other parts of the MLOps stack.

The Hugging Face/DVC setup is primarily for **static assets** (training/testing), but the "Data Engineering" discipline covers both the **offline factory** (training) and the **online delivery** (inference).

### Training & Testing (Offline)

You have labelled data. These are mainly hashes. To turn them into a GNN model, we need the **graph structure** of those transactions.

The dataset doesn't contain the transfer logs or internal traces;
At the Ingestion step we must pull in historical data from a full or **Archive Node** for each hash.

Once we have these traces, they get converted into feature graphs, we DVC track them. We don't want to call the node again every time we restart training. We version these "Ready-to-Train" graphs so training is fully repeatable without a validating node running.

### Inference (Online)

We eventually needs to detect MEV in the live **mempool**, that is before the beacon finalizes the block.

The Ingestion step at inference is a "Live Stream" of transactions.
In that scenario, **latency matters**. If the pipeline takes 2 seconds to fetch a trace and the GNN takes less than 200 ms to predict, given a block time of 12 seconds, we succeeded. If it takes over 12 seconds, the model is useless for real-time predictions.

Benchmark transactions per seconds: By measing the TPS during training data preparation, we already validated whether the ETL process can keep up with the Ethereum network's speed. Because the same process will execute for inference.

| Aspect | Training Data (Offline) | Inference Data (Online) |
| --- | --- | --- |
| **Source** | Labelled data + Node | Live Mempool + Full Node |
| **Storage** | DVC + Hugging Face | In-memory Cache (KV store/RAM) |
| **Goal** | **Reproducibility:** Can I get the same graph twice? | **Freshness:** Is this graph from the last 100ms? |
| **Engineering Task** | Batch Processing (ETL) | Stream Processing |

### Distinctions

1. **The Historical Pipeline:** (Dataset -> Node Fetch -> DVC -> Training). This is what is done with labelled data.

2. **The Real-Time Pipeline:** (Mempool -> Node Fetch -> GNN -> Detection). This uses some shared process, while it requires different engineering constraints (TPS and Latency) we measure before training.

## 📚 References

* **Medallion Architecture:** [Databricks's Glossary](https://www.databricks.com/glossary/medallion-architecture) - Organizing data into Bronze, Silver, and Gold.
* **DVC:** [dvc.org](https://dvc.org/) - A Git-based data versioning tool.
* **GNN Data Structures:** [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/) - Handling `Data` and `Batch` objects.
* **ETL Principles:** [Extract, Transform, Load](https://en.wikipedia.org/wiki/Extract,_transform,_load) - a standard for data movement and manipulation.