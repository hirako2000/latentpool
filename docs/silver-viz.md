# Data Observability - Silver Layer

The **Silver Layer** represents our "Graph-Ready" data. Before we move on, we use these visualizations to ensure the structural integrity and learnability of our transaction graphs.

## 🏁 Purpose

We don't just "ingest" data; we validate it. These diagnostics ensure that:

1. **Connectivity exists:** Every transaction forms a meaningful graph, not just isolated nodes.
2. **Features are dense:** We aren't training on empty or skewed feature sets.
3. **Complexity is controlled:** We identify "Black Hole" outliers that could swamp or even crash GPU memory.

## 📈 Diagnostic Suite

| Module | Primary Metric | Objective | Visualization |
| --- | --- | --- | --- |
| **Graph Topology** | Complexity Ratio | Verify "multi-hop" logic and reuse of tokens in swaps. | Nodes vs. Edges Scatter |
| **Path Depth** | Max Path Length | Ensure swap sequences () aren't flattened to 1-hop. | Depth Histogram |
| **Node Diversity** | Degree Distribution | Confirm we aren't over-reliant on a single hub (e.g., WETH). | Log-Log Degree Plot |
| **Feature Health** | Value Sparsity | Analyze -value transfers vs. high-liquidity moves. | Log-scale Heatmap |
| **Temporal Flow** | Burstiness Index | Identify competitive MEV clusters vs. "Normal" background noise. | Time-Series Heatmap |

## 🔍 Metric Deep Dives

### 📊 1. Graph Topology

Defined in [`viz_structure.py`](https://www.google.com/search?q=../src/latentpool/data/visualization/graph_structure.py).

* **Complexity Ratio:** Calculated as . A ratio  confirms tokens are reused within a transaction, preserving the "DNA" of MEV.
* **The 99th Percentile:** We use this value to set a hard "Edge Cap" for training batches to protect VRAM.

### 🏗️ Graph Path Depth

* **Path Length ():** MEV attacks like Sandwiches have specific signatures (e.g., 3-4 hops).
* **The Goal:** If the max path is consistently , it indicates a loss of sequence data during transformation.

### 🧬 Node Degree Diversity

* **The Hub Problem:** In DeFi, most edges touch `WETH`. We monitor this to ensure the GNN doesn't ignore exotic tokens.
* **The "Long Tail":** Healthy graph data should show a wide variety of token interactions.

### 🌡️ Feature Sparsity & Value Distribution

* **Zero-Value Transfers:** Arbitrage often involves internal contract logic with  value. We need to know if the GNN should prioritize "Path" over "Amount."


When we analyzed 1,058,098 edges to check the economic weight of the transfers:

* **Zero-Value Ratio:** 9.23%. High signal density; most edges represent actual movement.
* **Dynamic Range:** Values range from $1$ (Wei) to $10^{77}$ (Max uint256).

Due to the astronomical range, features MUST be processed via **Symmetric Log Scaling** ($sgn(x) \cdot \log(|x| + 1)$) to prevent gradient explosion. But that's for training, Gold parquets are the source of truth, we keep them as is.

_Opportunity_: We could determine the decimal by looking up the mapping for each ERC20 token, and convert correctly. It could serve for human readable analytics, but this wouln't help the GNN which purpose is to detect MEVs.

### 🕒 Temporal Burstiness

* **Competitive Clusters:** MEV isn't flat; it happens in bursts. This validates that our sampling of "Normal" transactions reflects real-world block timing.

## 🛠 How to run

see [justfile](../justfile)

### 🧠 Reflections on the "Black Hole"

Initial silver diagnostics revealed that while our **Average Edges** is a lean **3.59**, we have extreme outliers up to **2,500**.