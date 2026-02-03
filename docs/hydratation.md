# Metadata to Materialized Features

## Bronze Data

In this project, we saved ourselves from the initial "Mining" phase of Data Engineering.

We started with **Bronze Level** data—existing CSVs containing labels for both **Sandwich** and **Arbitrage** attacks. While we didn't have to label this ourselves, this data was "dehydrated": it contained the *who* and the _when_, but lacked the _how_

## Data Hydration

**Hydration** is the term for taking a sparse data point (such as a Transaction Hash) and filling it with further relevant context. In this case from an Ethereum Archive Node.

### GNNs

Graph Neural Networks (GNNs) learn from the **relationships** between entities.

The labeled data only said transactions happened, they were arbitrage, or sandwish attacks.
For the relationships, we need the **Receipt:**. They provides the "Logs", breadcrumbs of the transaction. These logs show which liquidity pools were touched, in what order, and with what volume etc.

Without hydration, we have a list; with hydration, we have  can build a **graph**.

## Engineering the pipeline

No without challanges, dealing with pandas.

### Structural

The "Arbitrage" labeled data utilized contains non-standard format where the `platforms` column contained internal commas, or nothing at all. Standard parsers (like the C-engine in Pandas) misidentified certain rows.

### Pattern-Matching Extraction

Instead of relying on the file's structure, we treated the file as a raw text stream. Used regex to identify the specific hexadecimal pattern of an Ethereum transaction hash. This ensured valid data without having to clean up the source.

### Bottlenecks

Fetching 124,000+ receipts from a full node isn't an issue, but setting up an archive node and all the rest... a bit of Alchemy saved us from having to setup an archive node locally. I have some spare 34TB of storage but downloading terabytes isn't my thing.

It took about 15 minutes to get all the transactions' receipts.

## Storage

The hydrated data, becomes, the **Silver Layer**:

| what | how |
| --- | --- |
| **Format** | Individual JSON files |
| **Naming Convention** | `{tx_hash}.json` |
| **Logic** | Enables **Lazy Loading**; data is pulled into memory, but we can easily batch process.


* **Bronze Data:** Raw, unprocessed data as it was found
* **Silver Data:** Cleaned, augmented, and joined data ready for analysis.

We track both, the bronze and silver data. Versioned. We can trace back if we discover the data augmentation wasn't right.

### Silver

We have the JSONs, but to train a GNN, we can't eat up 124,000 separate files. At least not efficiently.

TODO, parquets...