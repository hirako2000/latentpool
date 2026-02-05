## Discriminative Pivot

A transition from a "majority-guesser" to a model that can actually distinguish structural signatures.

### 1. Architectural Evolution: Depth & Stability

Moved form shallow network to a **4-Layer Deep Aggregation**. Allows the GNN to "hop" four steps across the graph, capturing more neighborhood context.

**Code Change (`sage.py`):**

```python
# BEFORE (Shallow)
self.conv1 = SAGEConv(in_channels, hidden_channels)
self.conv2 = SAGEConv(hidden_channels, out_channels)

# AFTER (Deep + Bottleneck)
self.conv1 = SAGEConv(in_channels, 512)
self.conv2 = SAGEConv(512, 512)
self.conv3 = SAGEConv(512, 512)
self.conv4 = SAGEConv(512, 512)
self.lin1 = Linear(1024, 128) # Bottleneck for distillation
```

Moved beyond 32% baseline accuracy to **62.8%**, successfully separating "Normal" transactions from "MEV" blobs.

### Finding the Sandwich

Initial runs saw Class 1 (Arbitrage) swallow the entire MEV category. So we use **Weighted Cross-Entropy** to get the model to identify the Sandwich class.

#### Arbitrage Dominance

We first tried to force Class 1 out of hiding.
**Code Change (`train.py`):**

```python
# BEFORE
criterion = nn.CrossEntropyLoss()
# AFTER
class_weights = torch.tensor([1.0, 10.0, 1.0])
criterion = nn.CrossEntropyLoss(weight=class_weights)

```

**1.00 Recall for Arbitrage**, but **0.00 for Sandwich**. The model called *every* MEV an Arbitrage to avoid the high penalty.

#### 5/5 Split

We leveled the weights and added **Label Smoothing** to reduce over-confidence.
**Code Change (`train.py`):**

```python
# BEFORE
class_weights = torch.tensor([1.0, 10.0, 1.0])
# AFTER
class_weights = torch.tensor([1.0, 5.0, 5.0])
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

```

First Sandwich detection! Precision for Sandwich hit **0.68**, proving the signature exists.

#### Further Sandwich hunting

Pivoted the weights heavily toward Class 2.
**Code Change (`train.py`):**

```python
# BEFORE
class_weights = torch.tensor([1.0, 5.0, 5.0])
# AFTER
class_weights = torch.tensor([1.0, 4.0, 12.0])

```

**0.99** Recall for Sandwich. Properly flagging the "MEV blob" as Sandwich instead of Arbitrage.

### Metric Comparison

| Metric | v2.0 (Shallow) | v2.1 (Arb-Heavy) | v2.2 (Sandwich-Heavy) |
| --- | --- | --- | --- |
| **Normal F1** | 0.90 | 0.90 | **0.93** |
| **Arb Recall** | 0.00 | **1.00** | 0.04 |
| **Sandwich Recall** | 0.00 | 0.00 | **0.99** |
| **Macro Avg F1** | 0.30 | 0.51 | **0.57** |


The model is now pretty good at **Noise Filtering** (Class 0). However, the oscillation between 1.00 Arb recall and 1.00 Sandwich recall confirms we have hit some **Feature Ceiling**. The model sees "MEV" clearly, but without internal call traces, Arbitrage and Sandwich occupy the same mathematical space.