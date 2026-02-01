# 🔗 DVC + Hugging Face

## Data Stack

We couple **DVC + Hugging Face** to provide a "Git for Data." Heavy artifacts (raw traces) are stored on Hugging Face while tracking their "pointers" in Git. This allows attributing every model prediction to a specific dataset version, necessary for **xAI**.

## Environment Configuration

DVC is handled as a project dependency. To enable Hugging Face support, we use the `dvc-hf` extension.

No need to install the standalone dvc and huggingface clients.

## Workflow

Using [just](../justfile) to for the DVC lifecycle to prevents "command fatigue" but specific uv and git commands can be executed.

```bash
# Initialize DVC in the repo (ran once) 
# DO NOT RUN THIS on this project
data-init:
    uv run dvc init --no-scm -f
    uv run dvc remote add -d hf_remote hf://hirako2000/latentpool-data
    git add .dvc/config
    git commit -m "Initialize DVC with Hugging Face remote"

# Add and version a new data batch 
# e.g using: `just data-add data/raw/batch_1.csv`
data-add path:
    uv run dvc add {{path}}
    uv run dvc push
    git add {{path}}.dvc data/.gitignore
    git commit -m "data: versioning batch {{path}}"
    # where {{path}} would be a file e.g data/raw/batch_1.csv

# To pull all data managed by DVC
data-pull:
    uv run dvc pull
```

## Data Lifecycle

We start with **Ingestion**. When a new `labeled_arbitrage.csv` gets in, it represents the "Bronze" asset.

Then comes **Versioning**. Running `just data-add` moves the file to the local DVC cache and uploads it to Hugging Face. Git only sees a small `.dvc` text file containing a hash (e.g., `md5: 22a1a2...`).

For **Traceability**, if we find a bug in the model down the road, we can `git checkout` an old commit. Running `just data-pull` will then fetch the *exact* version of the data that was used for that specific model.

```text
[ LOCAL MACHINE ]           [ GIT REPO ]            [ HUGGING FACE ]
  -----------------           ------------            ----------------
   raw_data      <-------+   (Pointer)       +---->  [  Artifact ]
  (MB Assets)            | raw_data.csv.dvc |    
                              ( Hash )
```

## Benchmarking

Transformation latency: We measure the time `uv run` takes to process a batch from the DVC cache into GNN features.

## 📚 References

* **DVC Remotes:** [DVC.org](https://doc.dvc.org/command-reference/remote/add) - Configuring cloud and hub-based storage.
* **Data Lineage:** See [Wikipedia](https://en.wikipedia.org/wiki/Data_lineage) - The importance of tracking data origins for ML.