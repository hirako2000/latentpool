

## Why GraphSAGE?

When we set out to classify MEV, both sanwdish attacks and arbitrage

MEV is a game of relationships: who sent what to which pool, and in what order.

We chose **GraphSAGE** (standing for *Sample and Aggregate*). It looks at the transaction's neighborhood. By aggregating features from adjacent nodes (tokens and addresses), the model builds a "structural fingerprint" of the behavior. An Arbitrage looks like a closed-loop circuit; a Sandwich looks like a predatory wrap-around. This type of GNN allows us to learn these topological signatures and apply them to transactions the model has never seen before.

| Concept | Traditional ML | GraphSAGE (GNN) |
| --- | --- | --- |
| **Data Focus** | Independent rows | Nodes and Edges (Relationships) |
| **Context** | Single point features | Neighborhood "structural" context |
| **Scalability** | High | High (via inductive learning) |


## The "Million Small Files" Problem

Our first attempt at training hit a wall before it even started. We had nearly 300,000 processed tensors. When we tried to load them using a standard Python list, it hung. Despite having enough ram. The interpreter struggled to manage the metadata for 300,000 separate objects sequentially.

To improve this process, we moved to a **Parallel In-Memory Loader**. By utilizing a `ThreadPoolExecutor`, we saturated the SSD I/O, reading 8 files simultaneously. This transformed a few-minutes "black hole" into a 50-second process with a live progress bar. We also implemented **Batching**, breaking the massive dataset into smaller groups of 256. This allowed the GPU (MPS which we tested with) to make frequent, incremental updates to the model weights rather than trying to digest the entire blockchain history in one go.

| Technical Choice | Purpose | Result |
| --- | --- | --- |
| **Parallel Loading** | Overcome Python I/O latency | ~5,300 files/sec loading speed |
| **In-Memory Storage** | Avoid slow disk reads during training | Near-instant batch delivery to GPU |
| **Mini-Batching** | Incremental weight updates | Stable gradient descent |


## Learning to Guess, Not to See

Once the data was flowing, we encountered a **Model Collapse**. We observed the "Training Loss" dropping: a sign the model was learning, but the "Test Accuracy" stayed frozen at roughly 32%.

Upon closer inspection of the prediction distribution, we found the model was simply guessing "Class 0" for every single transaction. In a 3-class dataset where one class makes up 32% of the data, this "lazy" strategy yields a 32% accuracy. The model probably found a mathematical shortcut: by predicting the majority, it could lower the loss without actually following the graph structure. This usually happens when raw features (like transaction values in Wei) are so massive that they "blind" the neural network, making the subtle structural signals invisible.


## The Refinement: Stability and Scaling

To break the collapse, we introduced **Input BatchNorm** and **Weight Decay**.

BatchNorm act as a "translator" that takes features of wildly different scales and puts them on a level playing field. So that a feature with a value of  doesn't carry  times more weight than a binary "is_source" flag. When we paired this with a lower learning rate and L2 regularization (Weight Decay), the model was forced to stop taking shortcuts.

The result was immediatly better. In just one epoch, the accuracy jumped from 32% to 62.89%. For the first time, the model began to discriminate between "Normal" transactions and "MEV" structures. It stopped guessing and spotting some actual distinctions.

| Refinement | Educational Analogy | Impact |
| --- | --- | --- |
| **Input BatchNorm** | Leveling the playing field | Prevents feature dominance |
| **Weight Decay** | Discouraging "obsession" | Prevents overfitting to outliers |
| **Reduced LR** | Taking smaller, careful steps | Prevents the model from "flying off" the map |
